import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.general_recommender.ncl import NCL
from recbole.utils import init_seed
from scipy.stats import binomtest, wilcoxon

from asym_model.model import AsymLightGCN
from asym_model.utils import setup_utf8_stdio
from final_full_graph_config import DATASET_SPECS, build_base_config


TOPKS = (10, 50)
METRICS = ("recall", "ndcg")

AMAZON_MIN5_CHECKPOINTS = {
    42: {
        "LightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-04-2026_23-39-32.pth",
        "AsymLightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-04-2026_23-39-43.pth",
        "NCL": "saved/final_reduced/clean_amazon_reduced_min5/NCL-May-08-2026_02-44-26.pth",
    },
    43: {
        "LightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-05-2026_20-11-53.pth",
        "AsymLightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-05-2026_20-12-04.pth",
        "NCL": "saved/final_reduced/clean_amazon_reduced_min5/NCL-May-08-2026_02-45-08.pth",
    },
    44: {
        "LightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-05-2026_20-13-35.pth",
        "AsymLightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-05-2026_20-13-45.pth",
        "NCL": "saved/final_reduced/clean_amazon_reduced_min5/NCL-May-08-2026_02-45-50.pth",
    },
    45: {
        "LightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-08-2026_02-37-32.pth",
        "AsymLightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-08-2026_02-37-43.pth",
        "NCL": "saved/final_reduced/clean_amazon_reduced_min5/NCL-May-08-2026_02-46-31.pth",
    },
    46: {
        "LightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-08-2026_02-38-21.pth",
        "AsymLightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-08-2026_02-38-31.pth",
        "NCL": "saved/final_reduced/clean_amazon_reduced_min5/NCL-May-08-2026_02-47-13.pth",
    },
    47: {
        "LightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-08-2026_02-38-49.pth",
        "AsymLightGCN": "saved/final_reduced/clean_amazon_reduced_min5/LightGCN-May-08-2026_02-38-59.pth",
        "NCL": "saved/final_reduced/clean_amazon_reduced_min5/NCL-May-08-2026_02-47-54.pth",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-user full-sort metrics and paired user-level tests "
            "for paper-facing comparisons."
        )
    )
    parser.add_argument("--dataset", default="amazon_reduced_min5", choices=["amazon_reduced_min5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47])
    parser.add_argument("--models", nargs="+", default=["LightGCN", "AsymLightGCN", "NCL"])
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["Semantic-only", "MostPopIndependent"],
        choices=["Semantic-only", "MostPopIndependent"],
    )
    parser.add_argument("--topks", nargs="+", type=int, default=list(TOPKS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--eval-step", type=int, default=15)
    parser.add_argument("--stopping-step", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260514)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def make_config(spec, seed, args, model_name, extra=None):
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
    config_dict["model"] = model_name
    config_dict["dataset"] = spec["dataset_name"]
    if extra:
        config_dict.update(extra)
    return Config(
        model=model_name,
        dataset=spec["dataset_name"],
        config_file_list=["config.yaml"],
        config_dict=config_dict,
    )


def prepare_data(spec, seed, args):
    init_seed(seed, reproducibility=True)
    config = make_config(spec, seed, args, "LightGCN")
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return config, dataset, train_data, valid_data, test_data


def build_user_items(loader, uid_field, iid_field):
    user_items = defaultdict(set)
    users = loader.dataset.inter_feat[uid_field].cpu().numpy().tolist()
    items = loader.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    for user, item in zip(users, items):
        user_items[int(user)].add(int(item))
    return user_items


def union_user_items(*mappings):
    combined = defaultdict(set)
    for mapping in mappings:
        for user, items in mapping.items():
            combined[int(user)].update(int(item) for item in items)
    return combined


def dcg_at_k(recommended, relevant, topk):
    return sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:topk])
        if int(item) in relevant
    )


def metrics_for_recommendations(recommended, relevant, topks):
    row = {"relevant_count": len(relevant)}
    for topk in topks:
        rec_at_k = [int(item) for item in recommended[:topk]]
        hits = len(set(rec_at_k) & relevant)
        ideal_len = min(len(relevant), topk)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_len))
        row[f"recall@{topk}"] = hits / len(relevant) if relevant else 0.0
        row[f"ndcg@{topk}"] = dcg_at_k(rec_at_k, relevant, topk) / idcg if idcg > 0 else 0.0
    return row


def instantiate_model(kind, config, dataset):
    if kind == "LightGCN":
        return LightGCN(config, dataset)
    if kind == "AsymLightGCN":
        return AsymLightGCN(config, dataset)
    if kind == "NCL":
        return NCL(config, dataset)
    raise ValueError(f"Unknown model kind: {kind}")


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


def load_model(kind, checkpoint_path, spec, seed, args, dataset):
    if kind == "AsymLightGCN":
        config = make_config(
            spec,
            seed,
            args,
            "LightGCN",
            extra={"magnitude_calibration_gamma": 0.25},
        )
    elif kind == "NCL":
        config = make_config(
            spec,
            seed,
            args,
            "NCL",
            extra={
                "ssl_temp": 0.1,
                "ssl_reg": 1e-7,
                "hyper_layers": 1,
                "alpha": 1.0,
                "proto_reg": 1e-7,
                "num_clusters": 36,
                "m_step": 1,
                "train_batch_size": 256,
                "kmeans_device": "cpu",
                "niter": 5,
            },
        )
    else:
        config = make_config(spec, seed, args, "LightGCN")

    model = instantiate_model(kind, config, dataset)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return move_model_to_device(model, args.device)


def evaluate_torch_model_per_user(
    model,
    model_name,
    dataset_name,
    seed,
    target_sets,
    exclude_sets,
    uid_field,
    topks,
    batch_size,
    device,
):
    max_k = max(topks)
    eval_users = sorted(target_sets.keys())
    rows = []
    torch_device = torch.device(device)
    with torch.no_grad():
        for offset in range(0, len(eval_users), batch_size):
            batch_users = eval_users[offset : offset + batch_size]
            interaction = Interaction(
                {uid_field: torch.tensor(batch_users, dtype=torch.long, device=torch_device)}
            )
            scores = model.full_sort_predict(interaction).view(len(batch_users), -1)
            scores[:, 0] = -torch.inf
            for row_idx, user in enumerate(batch_users):
                excluded = [item for item in exclude_sets[user] if item < scores.shape[1]]
                if excluded:
                    scores[row_idx, torch.tensor(excluded, dtype=torch.long, device=torch_device)] = -torch.inf
            top_items = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()
            for row_idx, user in enumerate(batch_users):
                relevant = target_sets[user]
                metric_row = metrics_for_recommendations(top_items[row_idx], relevant, topks)
                metric_row.update(
                    {
                        "Dataset": dataset_name,
                        "Seed": seed,
                        "Model": model_name,
                        "User": int(user),
                    }
                )
                rows.append(metric_row)
    return rows


def evaluate_mostpop_per_user(dataset_name, seed, train_data, target_sets, exclude_sets, iid_field, topks):
    train_items = train_data.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    popularity = [int(item) for item, _ in Counter(train_items).most_common()]
    max_k = max(topks)
    rows = []
    for user in sorted(target_sets.keys()):
        relevant = target_sets[user]
        excluded = exclude_sets[user]
        recs = []
        for item in popularity:
            if item in excluded:
                continue
            recs.append(item)
            if len(recs) >= max_k:
                break
        metric_row = metrics_for_recommendations(recs, relevant, topks)
        metric_row.update(
            {
                "Dataset": dataset_name,
                "Seed": seed,
                "Model": "MostPopIndependent",
                "User": int(user),
            }
        )
        rows.append(metric_row)
    return rows


def load_semantic_item_matrix(spec, dataset, iid_field):
    raw_embeddings = torch.load(spec["semantic_embs_path"], map_location="cpu")
    first_embedding = next(iter(raw_embeddings.values()))
    raw_dim = int(torch.as_tensor(first_embedding, dtype=torch.float32).numel())
    item_semantics = torch.zeros((dataset.item_num, raw_dim), dtype=torch.float32)
    mapped = 0
    for token, iid in dataset.field2token_id[iid_field].items():
        if token == "[PAD]":
            continue
        embedding = raw_embeddings.get(str(token))
        if embedding is None:
            continue
        item_semantics[int(iid)] = torch.as_tensor(embedding, dtype=torch.float32)
        mapped += 1
    return item_semantics, mapped


def profile_from_train_items(item_semantics, items):
    item_ids = [int(item) for item in items if int(item) != 0]
    if not item_ids:
        return None
    profile = item_semantics[item_ids].mean(dim=0)
    norm = torch.linalg.vector_norm(profile)
    if not torch.isfinite(norm) or norm <= 0:
        return None
    return profile / norm


def evaluate_semantic_per_user(
    spec,
    dataset,
    dataset_name,
    seed,
    train_sets,
    target_sets,
    exclude_sets,
    iid_field,
    topks,
    batch_size,
    device,
):
    max_k = max(topks)
    item_semantics, mapped = load_semantic_item_matrix(spec, dataset, iid_field)
    if mapped == 0:
        raise RuntimeError(f"No semantic embeddings mapped for {dataset_name}")
    item_semantics_cpu = F.normalize(item_semantics, p=2, dim=1).cpu()
    item_semantics_device = item_semantics_cpu.to(device)
    item_semantics_t = item_semantics_device.transpose(0, 1).contiguous()
    eval_users = sorted(target_sets.keys())
    rows = []
    no_profile_users = 0
    for offset in range(0, len(eval_users), batch_size):
        batch_users = eval_users[offset : offset + batch_size]
        profiles = []
        kept_users = []
        for user in batch_users:
            profile = profile_from_train_items(item_semantics_cpu, train_sets[user])
            if profile is None:
                no_profile_users += 1
                continue
            profiles.append(profile)
            kept_users.append(user)
        if not profiles:
            continue
        profile_tensor = torch.stack(profiles).to(device)
        scores = torch.matmul(profile_tensor, item_semantics_t)
        scores[:, 0] = -torch.inf
        for row_idx, user in enumerate(kept_users):
            excluded = [item for item in exclude_sets[user] if item < scores.shape[1]]
            if excluded:
                scores[row_idx, torch.tensor(excluded, dtype=torch.long, device=device)] = -torch.inf
        top_items = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()
        for row_idx, user in enumerate(kept_users):
            relevant = target_sets[user]
            metric_row = metrics_for_recommendations(top_items[row_idx], relevant, topks)
            metric_row.update(
                {
                    "Dataset": dataset_name,
                    "Seed": seed,
                    "Model": "Semantic-only",
                    "User": int(user),
                    "No Profile Users": no_profile_users,
                }
            )
            rows.append(metric_row)
    return rows


def summarize_per_user(per_user):
    records = []
    for (dataset_name, seed, model), group in per_user.groupby(["Dataset", "Seed", "Model"]):
        record = {"Dataset": dataset_name, "Seed": int(seed), "Model": model, "Users": len(group)}
        for topk in TOPKS:
            for metric in METRICS:
                col = f"{metric}@{topk}"
                if col in group.columns:
                    record[col] = float(group[col].mean())
        records.append(record)
    return pd.DataFrame(records).sort_values(["Dataset", "Seed", "Model"])


def bootstrap_ci(values, samples, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_test_table(per_user, comparisons, topks, bootstrap_samples, bootstrap_seed):
    records = []
    for dataset_name, target_model, baseline_model in comparisons:
        for topk in topks:
            for metric in METRICS:
                col = f"{metric}@{topk}"
                left = per_user[
                    (per_user["Dataset"] == dataset_name) & (per_user["Model"] == target_model)
                ][["Seed", "User", col]]
                right = per_user[
                    (per_user["Dataset"] == dataset_name) & (per_user["Model"] == baseline_model)
                ][["Seed", "User", col]]
                paired = left.merge(
                    right,
                    on=["Seed", "User"],
                    how="inner",
                    suffixes=("_target", "_baseline"),
                )
                if paired.empty:
                    continue
                paired["delta"] = paired[f"{col}_target"] - paired[f"{col}_baseline"]
                user_deltas = paired.groupby("User")["delta"].mean().to_numpy(dtype=np.float64)
                nonzero = user_deltas[np.abs(user_deltas) > 1e-12]
                positive = int((user_deltas > 1e-12).sum())
                negative = int((user_deltas < -1e-12).sum())
                zero = int((np.abs(user_deltas) <= 1e-12).sum())
                ci_low, ci_high = bootstrap_ci(
                    user_deltas,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + len(records),
                )
                if nonzero.size:
                    try:
                        wilcoxon_p = float(
                            wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue
                        )
                    except ValueError:
                        wilcoxon_p = math.nan
                    sign_p = float(binomtest(positive, positive + negative, 0.5).pvalue)
                else:
                    wilcoxon_p = math.nan
                    sign_p = math.nan
                records.append(
                    {
                        "Dataset": dataset_name,
                        "Comparison": f"{target_model} - {baseline_model}",
                        "Metric": col,
                        "Paired Seed-User Rows": int(len(paired)),
                        "Users": int(user_deltas.size),
                        "Mean Delta": float(user_deltas.mean()),
                        "Median Delta": float(np.median(user_deltas)),
                        "Bootstrap 95% CI Low": ci_low,
                        "Bootstrap 95% CI High": ci_high,
                        "Wilcoxon p": wilcoxon_p,
                        "Sign p": sign_p,
                        "Positive Users": positive,
                        "Negative Users": negative,
                        "Zero Users": zero,
                    }
                )
    return pd.DataFrame(records)


def format_p(value):
    if pd.isna(value):
        return "NA"
    if value < 1e-4:
        return "<0.0001"
    return f"{value:.4f}"


def write_markdown(path, summary, tests, args):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# User-Level Significance Analysis\n\n")
        handle.write(
            "Per-user Recall@K and NDCG@K were recomputed with full-sort candidate "
            "masking for the paper-facing Amazon min5 test split. Deltas are averaged "
            "per user across available seeds before bootstrap and non-parametric tests.\n\n"
        )
        handle.write(f"Bootstrap samples: {args.bootstrap_samples}\n\n")
        handle.write("## Aggregate Check\n\n")
        handle.write(summary.to_markdown(index=False, floatfmt=".6f"))
        handle.write("\n\n## Paired User-Level Tests\n\n")
        printable = tests.copy()
        for col in ["Mean Delta", "Median Delta", "Bootstrap 95% CI Low", "Bootstrap 95% CI High"]:
            printable[col] = printable[col].map(lambda value: f"{value:.6f}")
        for col in ["Wilcoxon p", "Sign p"]:
            printable[col] = printable[col].map(format_p)
        handle.write(printable.to_markdown(index=False))
        handle.write("\n")


def main():
    setup_utf8_stdio()
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")

    spec = DATASET_SPECS[args.dataset]
    dataset_name = spec["dataset_name"]
    if args.output_prefix is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_prefix = Path("train_logs") / f"user_level_significance_{timestamp}"
    else:
        output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    manifest = {"dataset": args.dataset, "dataset_name": dataset_name, "seeds": args.seeds, "checkpoints": {}}

    for seed in args.seeds:
        started = time.perf_counter()
        print(f"Preparing {dataset_name} seed={seed}", flush=True)
        ref_config, dataset, train_data, valid_data, test_data = prepare_data(spec, seed, args)
        uid_field = ref_config["USER_ID_FIELD"]
        iid_field = ref_config["ITEM_ID_FIELD"]
        train_sets = build_user_items(train_data, uid_field, iid_field)
        valid_sets = build_user_items(valid_data, uid_field, iid_field)
        test_sets = build_user_items(test_data, uid_field, iid_field)
        exclude_sets = union_user_items(train_sets, valid_sets)

        seed_checkpoints = AMAZON_MIN5_CHECKPOINTS[seed]
        manifest["checkpoints"][str(seed)] = seed_checkpoints
        for model_name in args.models:
            checkpoint_path = Path(seed_checkpoints[model_name])
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            print(f"  Evaluating {model_name}: {checkpoint_path.name}", flush=True)
            model = load_model(model_name, checkpoint_path, spec, seed, args, train_data.dataset)
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
        print(f"Finished seed={seed} in {time.perf_counter() - started:.1f}s", flush=True)

    per_user = pd.DataFrame(all_rows)
    per_user_path = output_prefix.with_name(f"{output_prefix.name}_per_user.csv")
    per_user.to_csv(per_user_path, index=False)

    summary = summarize_per_user(per_user)
    summary_path = output_prefix.with_name(f"{output_prefix.name}_aggregate_check.csv")
    summary.to_csv(summary_path, index=False)

    comparisons = [
        (dataset_name, "AsymLightGCN", "LightGCN"),
        (dataset_name, "AsymLightGCN", "NCL"),
        (dataset_name, "AsymLightGCN", "Semantic-only"),
        (dataset_name, "AsymLightGCN", "MostPopIndependent"),
    ]
    tests = paired_test_table(
        per_user=per_user,
        comparisons=comparisons,
        topks=args.topks,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    tests_path = output_prefix.with_name(f"{output_prefix.name}_paired_tests.csv")
    tests.to_csv(tests_path, index=False)

    manifest_path = output_prefix.with_name(f"{output_prefix.name}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md_path = output_prefix.with_suffix(".md")
    write_markdown(md_path, summary, tests, args)

    print(f"Saved per-user metrics to {per_user_path}", flush=True)
    print(f"Saved aggregate check to {summary_path}", flush=True)
    print(f"Saved paired tests to {tests_path}", flush=True)
    print(f"Saved report to {md_path}", flush=True)


if __name__ == "__main__":
    main()
