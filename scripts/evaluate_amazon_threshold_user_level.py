import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from recbole.config import Config
from recbole.model.general_recommender import BPR

from asym_model import SemanticGatedBPR
from asym_model.model import AsymLightGCN
from evaluate_user_level_significance import (
    build_user_items,
    evaluate_mostpop_per_user,
    evaluate_semantic_per_user,
    evaluate_torch_model_per_user,
    paired_test_table,
    prepare_data,
    summarize_per_user,
    union_user_items,
)
from final_full_graph_config import DATASET_SPECS, build_base_config
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.general_recommender.ncl import NCL


DEFAULT_DATASETS = [
    "amazon_reduced_min4",
    "amazon_reduced_min6",
    "amazon_reduced_min8",
    "amazon_reduced_min10",
]
CHECKPOINT_MODELS = ["BPR", "SemanticGatedBPR", "LightGCN", "NCL", "AsymLightGCN"]
CONTROL_MODELS = ["Semantic-only", "MostPopIndependent"]
TOPKS = [10, 50]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-user paired tests for the Amazon min-degree threshold sweep."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--models", nargs="+", default=CHECKPOINT_MODELS, choices=CHECKPOINT_MODELS)
    parser.add_argument("--controls", nargs="+", default=CONTROL_MODELS, choices=CONTROL_MODELS)
    parser.add_argument("--topks", nargs="+", type=int, default=TOPKS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--eval-step", type=int, default=15)
    parser.add_argument("--stopping-step", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260514)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def has_semantic_state(state_dict):
    semantic_markers = ("semantic", "centroid", "gating", "log_vars")
    return any(any(marker in key.lower() for marker in semantic_markers) for key in state_dict)


def checkpoint_seed(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        return None
    return int(config["seed"])


def classify_core_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"Checkpoint has no config: {path}")
    seed = int(config["seed"])
    if path.name.startswith("NCL-"):
        return seed, "NCL"
    state_dict = checkpoint["state_dict"]
    if has_semantic_state(state_dict):
        return seed, "AsymLightGCN"
    return seed, "LightGCN"


def discover_checkpoints(spec):
    checkpoints = {}
    core_dir = Path(spec["checkpoint_dir"])
    for path in core_dir.glob("*.pth"):
        seed, model = classify_core_checkpoint(path)
        previous = checkpoints.get((seed, model))
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            checkpoints[(seed, model)] = path

    extra_root = Path("saved/extra_reduced_baselines") / spec["dataset_name"]
    for model in ["BPR", "SemanticGatedBPR"]:
        model_dir = extra_root / model
        for path in model_dir.glob("*.pth"):
            seed = checkpoint_seed(path)
            previous = checkpoints.get((seed, model))
            if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
                checkpoints[(seed, model)] = path
    return checkpoints


def make_config(spec, seed, args, model_name, checkpoint_path=None):
    config_dict = build_base_config(
        spec,
        seed=seed,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_step=args.eval_step,
        stopping_step=args.stopping_step,
    )
    config_dict["use_gpu"] = args.device == "cuda"
    if args.device == "cpu":
        config_dict["gpu_id"] = ""

    if model_name == "AsymLightGCN":
        recbole_model = "LightGCN"
        config_dict["magnitude_calibration_gamma"] = 0.25
    elif model_name == "SemanticGatedBPR":
        recbole_model = "BPR"
    else:
        recbole_model = model_name

    if model_name == "NCL" and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ck_config = checkpoint.get("config")
        if ck_config is not None:
            for key in [
                "ssl_temp",
                "ssl_reg",
                "hyper_layers",
                "alpha",
                "proto_reg",
                "num_clusters",
                "m_step",
                "niter",
                "kmeans_device",
            ]:
                try:
                    config_dict[key] = ck_config[key]
                except Exception:
                    pass

    config_dict["model"] = recbole_model
    config_dict["dataset"] = spec["dataset_name"]
    return Config(
        model=recbole_model,
        dataset=spec["dataset_name"],
        config_file_list=["config.yaml"],
        config_dict=config_dict,
    )


def instantiate_model(model_name, config, dataset):
    if model_name == "LightGCN":
        return LightGCN(config, dataset)
    if model_name == "AsymLightGCN":
        return AsymLightGCN(config, dataset)
    if model_name == "NCL":
        return NCL(config, dataset)
    if model_name == "BPR":
        return BPR(config, dataset)
    if model_name == "SemanticGatedBPR":
        return SemanticGatedBPR(config, dataset)
    raise ValueError(model_name)


def move_model_to_device(model, device):
    torch_device = torch.device(device)
    model.to(torch_device)
    model.device = torch_device
    for attr_name in ("norm_adj_matrix", "norm_adj_mat"):
        if hasattr(model, attr_name):
            value = getattr(model, attr_name)
            if value is not None:
                setattr(model, attr_name, value.to(torch_device))
    for attr_name in ("restore_user_e", "restore_item_e"):
        if hasattr(model, attr_name):
            setattr(model, attr_name, None)
    model.eval()
    return model


def load_checkpoint_model(model_name, checkpoint_path, spec, seed, args, dataset):
    config = make_config(spec, seed, args, model_name, checkpoint_path=checkpoint_path)
    model = instantiate_model(model_name, config, dataset)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return move_model_to_device(model, args.device)


def paired_comparisons(dataset_names, models, controls):
    baselines = ["LightGCN", "NCL", "SemanticGatedBPR", "Semantic-only", "MostPopIndependent"]
    available = set(models) | set(controls)
    records = []
    for dataset_name in dataset_names:
        for baseline in baselines:
            if baseline in available:
                records.append((dataset_name, "AsymLightGCN", baseline))
    return records


def format_p(value):
    if pd.isna(value):
        return "NA"
    if value < 1e-4:
        return "<0.0001"
    return f"{float(value):.4f}"


def write_markdown(path, summary, tests, manifest, args):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Amazon Threshold User-Level Significance\n\n")
        handle.write(
            "Per-user Recall@K and NDCG@K were recomputed with full-sort candidate masking "
            "for Amazon min4/min6/min8/min10. Deltas are averaged per user across available "
            "seeds before bootstrap and paired non-parametric tests.\n\n"
        )
        handle.write(f"Bootstrap samples: {args.bootstrap_samples}\n\n")
        handle.write("## Aggregate Check\n\n")
        handle.write(summary.to_markdown(index=False, floatfmt=".6f"))
        handle.write("\n\n## Paired User-Level Tests\n\n")
        printable = tests.copy()
        for col in ["Mean Delta", "Median Delta", "Bootstrap 95% CI Low", "Bootstrap 95% CI High"]:
            printable[col] = printable[col].map(lambda value: f"{float(value):.6f}")
        for col in ["Wilcoxon p", "Sign p"]:
            printable[col] = printable[col].map(format_p)
        handle.write(printable.to_markdown(index=False))
        handle.write("\n\n## Checkpoints\n\n")
        for dataset_key, dataset_manifest in manifest["checkpoints"].items():
            handle.write(f"### {dataset_key}\n\n")
            for seed, seed_rows in sorted(dataset_manifest.items(), key=lambda item: int(item[0])):
                handle.write(f"- seed {seed}: ")
                handle.write("; ".join(f"{model}={Path(path).name}" for model, path in sorted(seed_rows.items())))
                handle.write("\n")


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")

    if args.output_prefix is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_prefix = Path("train_logs") / f"amazon_threshold_user_level_{timestamp}"
    else:
        output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    manifest = {
        "datasets": args.datasets,
        "seeds": args.seeds,
        "models": args.models,
        "controls": args.controls,
        "checkpoints": {},
    }

    for dataset_key in args.datasets:
        spec = DATASET_SPECS[dataset_key]
        dataset_name = spec["dataset_name"]
        checkpoints = discover_checkpoints(spec)
        manifest["checkpoints"][dataset_key] = {}
        for seed in args.seeds:
            print(f"Preparing {dataset_name} seed={seed}", flush=True)
            ref_config, dataset, train_data, valid_data, test_data = prepare_data(spec, seed, args)
            uid_field = ref_config["USER_ID_FIELD"]
            iid_field = ref_config["ITEM_ID_FIELD"]
            train_sets = build_user_items(train_data, uid_field, iid_field)
            valid_sets = build_user_items(valid_data, uid_field, iid_field)
            test_sets = build_user_items(test_data, uid_field, iid_field)
            exclude_sets = union_user_items(train_sets, valid_sets)

            manifest["checkpoints"][dataset_key][str(seed)] = {}
            for model_name in args.models:
                checkpoint_path = checkpoints.get((seed, model_name))
                if checkpoint_path is None:
                    print(f"  Missing checkpoint for {model_name}; skipping", flush=True)
                    continue
                print(f"  Evaluating {model_name}: {checkpoint_path.name}", flush=True)
                manifest["checkpoints"][dataset_key][str(seed)][model_name] = str(checkpoint_path)
                model = load_checkpoint_model(model_name, checkpoint_path, spec, seed, args, train_data.dataset)
                all_rows.extend(
                    evaluate_torch_model_per_user(
                        model=model,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        seed=seed,
                        target_sets=test_sets,
                        exclude_sets=exclude_sets,
                        uid_field=uid_field,
                        topks=args.topks,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                )
                del model

            if "MostPopIndependent" in args.controls:
                print("  Evaluating MostPopIndependent", flush=True)
                all_rows.extend(
                    evaluate_mostpop_per_user(
                        dataset_name=dataset_name,
                        seed=seed,
                        train_data=train_data,
                        target_sets=test_sets,
                        exclude_sets=exclude_sets,
                        iid_field=iid_field,
                        topks=args.topks,
                    )
                )
            if "Semantic-only" in args.controls:
                print("  Evaluating Semantic-only", flush=True)
                all_rows.extend(
                    evaluate_semantic_per_user(
                        spec=spec,
                        dataset=dataset,
                        dataset_name=dataset_name,
                        seed=seed,
                        train_sets=train_sets,
                        target_sets=test_sets,
                        exclude_sets=exclude_sets,
                        iid_field=iid_field,
                        topks=args.topks,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                )

    per_user = pd.DataFrame(all_rows)
    per_user_path = output_prefix.with_name(f"{output_prefix.name}_per_user.csv")
    per_user.to_csv(per_user_path, index=False)

    summary = summarize_per_user(per_user)
    summary_path = output_prefix.with_name(f"{output_prefix.name}_aggregate_check.csv")
    summary.to_csv(summary_path, index=False)

    dataset_names = [DATASET_SPECS[key]["dataset_name"] for key in args.datasets]
    tests = paired_test_table(
        per_user=per_user,
        comparisons=paired_comparisons(dataset_names, args.models, args.controls),
        topks=args.topks,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    tests_path = output_prefix.with_name(f"{output_prefix.name}_paired_tests.csv")
    tests.to_csv(tests_path, index=False)

    manifest_path = output_prefix.with_name(f"{output_prefix.name}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md_path = output_prefix.with_suffix(".md")
    write_markdown(md_path, summary, tests, manifest, args)

    print(f"Saved per-user metrics to {per_user_path}")
    print(f"Saved aggregate check to {summary_path}")
    print(f"Saved paired tests to {tests_path}")
    print(f"Saved manifest to {manifest_path}")
    print(f"Saved report to {md_path}")


if __name__ == "__main__":
    main()
