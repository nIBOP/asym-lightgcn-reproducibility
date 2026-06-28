import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.general_recommender.ncl import NCL
from recbole.utils import init_seed

from asym_model.model import AsymLightGCN
from asym_model.utils import setup_utf8_stdio
from final_full_graph_config import DATASET_SPECS, build_base_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved reduced-dataset checkpoints by item-popularity strata."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "amazon_reduced",
            "amazon_reduced_min3",
            "amazon_reduced_min5",
            "yelp_reduced",
            "movies_reduced",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--eval-step", type=int, default=15)
    parser.add_argument("--stopping-step", type=int, default=2)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["valid", "test"])
    parser.add_argument("--topks", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--lightgcn-checkpoint", default=None)
    parser.add_argument("--asym-checkpoint", default=None)
    parser.add_argument("--ncl-checkpoint", default=None)
    parser.add_argument("--ncl-num-clusters", type=int, default=160)
    parser.add_argument("--ncl-kmeans-device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--ncl-kmeans-niter", type=int, default=5)
    parser.add_argument("--ncl-loss-micro-batch-size", type=int, default=256)
    parser.add_argument("--include-asym-gamma0", action="store_true")
    parser.add_argument(
        "--asym-gammas",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Evaluate AsymLightGCN checkpoints with these inference-time "
            "magnitude calibration gamma values. This is eval-only and does "
            "not retrain the checkpoint."
        ),
    )
    parser.add_argument(
        "--load-other-parameter",
        action="store_true",
        help="Load RecBole checkpoint other_parameter cache such as restore_user_e/restore_item_e.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Defaults to train_logs/reduced_item_strata_<timestamp>.",
    )
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def setup(seed):
    init_seed(seed, reproducibility=True)
    sys.argv = [sys.argv[0]]


def format_gamma(value):
    return f"{value:g}"


def unique_gammas(values):
    result = []
    seen = set()
    for value in values:
        key = round(float(value), 12)
        if key < 0:
            raise ValueError("--asym-gammas values must be non-negative")
        if key not in seen:
            seen.add(key)
            result.append(float(value))
    return result


def make_config(spec, args, model_name, extra=None):
    config_dict = build_base_config(
        spec,
        seed=args.seed,
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


def prepare_reference_data(spec, args):
    config = make_config(spec, args, "LightGCN")
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


def union_item_sets(left, right):
    combined = defaultdict(set)
    for user, items in left.items():
        combined[user].update(items)
    for user, items in right.items():
        combined[user].update(items)
    return combined


def build_item_strata(train_data, eval_loaders, uid_field, iid_field):
    train_items = train_data.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    train_item_counts = pd.Series(train_items).value_counts()
    total_train_items = len(train_item_counts)
    head_cutoff = max(1, int(total_train_items * 0.20))
    torso_cutoff = min(max(head_cutoff + 1, int(total_train_items * 0.50)), total_train_items)

    head_items = set(int(item) for item in train_item_counts.index[:head_cutoff])
    torso_items = set(int(item) for item in train_item_counts.index[head_cutoff:torso_cutoff])
    tail_items_train = set(int(item) for item in train_item_counts.index[torso_cutoff:])

    split_strata = {}
    train_item_set = set(int(item) for item in train_item_counts.index)
    for split_name, loader in eval_loaders.items():
        eval_items = set(
            int(item)
            for item in loader.dataset.inter_feat[iid_field].cpu().numpy().tolist()
        )
        cold_items = eval_items - train_item_set
        tail_items = tail_items_train | cold_items
        split_strata[split_name] = {
            "head": head_items,
            "torso": torso_items,
            "tail": tail_items,
            "non_head": torso_items | tail_items,
            "cold": cold_items,
        }

    return {
        "train_item_count": total_train_items,
        "head_item_count": len(head_items),
        "torso_item_count": len(torso_items),
        "tail_train_item_count": len(tail_items_train),
        "split_strata": split_strata,
    }


def instantiate_model(kind, config, dataset):
    if kind == "LightGCN":
        return LightGCN(config, dataset)
    if kind == "AsymLightGCN":
        return AsymLightGCN(config, dataset)
    if kind == "NCL":
        return NCL(config, dataset)
    raise ValueError(f"Unknown model kind: {kind}")


def move_model_to_eval_device(model, device):
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
            value = getattr(model, attr_name)
            if value is not None:
                setattr(model, attr_name, value.to(torch_device))
    return model


def load_model(kind, label, checkpoint_path, spec, model_dataset, args):
    if kind == "AsymLightGCN":
        config = make_config(
            spec,
            args,
            "LightGCN",
            extra={"magnitude_calibration_gamma": 0.25},
        )
    elif kind == "NCL":
        config = make_config(
            spec,
            args,
            "NCL",
            extra={
                "ssl_temp": 0.1,
                "ssl_reg": 1e-7,
                "hyper_layers": 1,
                "alpha": 1.0,
                "proto_reg": 1e-7,
                "num_clusters": args.ncl_num_clusters,
                "m_step": 1,
                "train_batch_size": args.ncl_loss_micro_batch_size,
                "kmeans_device": args.ncl_kmeans_device,
                "niter": args.ncl_kmeans_niter,
            },
        )
    else:
        config = make_config(spec, args, "LightGCN")

    model = instantiate_model(kind, config, model_dataset)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if unexpected:
        print(f"[{label}] Unexpected checkpoint keys: {unexpected}", flush=True)
    if missing:
        print(f"[{label}] Missing checkpoint keys: {missing}", flush=True)
    if args.load_other_parameter:
        model.load_other_parameter(checkpoint.get("other_parameter"))
    move_model_to_eval_device(model, args.device)
    model.eval()
    return config, model


def new_metric_sums(topks):
    return {
        topk: {
            "users": 0,
            "recall": 0.0,
            "recall_micro_num": 0,
            "recall_micro_den": 0,
            "ndcg": 0.0,
            "recall_head": 0.0,
            "recall_head_users": 0,
            "ndcg_head": 0.0,
            "ndcg_head_users": 0,
            "recall_torso": 0.0,
            "recall_torso_users": 0,
            "ndcg_torso": 0.0,
            "ndcg_torso_users": 0,
            "recall_tail": 0.0,
            "recall_tail_users": 0,
            "ndcg_tail": 0.0,
            "ndcg_tail_users": 0,
            "recall_nh": 0.0,
            "recall_nh_users": 0,
            "recall_nh_micro_num": 0,
            "recall_nh_micro_den": 0,
            "ndcg_nh": 0.0,
            "ndcg_nh_users": 0,
            "non_head_recommended": 0,
            "recommended": 0,
        }
        for topk in topks
    }


def dcg_at_k(recommended, relevant, topk):
    return sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:topk])
        if int(item) in relevant
    )


def update_segment_metrics(sums, topk, recommended, relevant, segment_items, prefix):
    target = relevant & segment_items
    if not target:
        return
    hits = len(set(int(item) for item in recommended[:topk]) & target)
    recall = hits / len(target)
    ideal_len = min(len(target), topk)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_len))
    ndcg = dcg_at_k(recommended, target, topk) / idcg if idcg > 0 else 0.0

    sums[f"recall_{prefix}"] += recall
    sums[f"recall_{prefix}_users"] += 1
    sums[f"ndcg_{prefix}"] += ndcg
    sums[f"ndcg_{prefix}_users"] += 1
    if prefix == "nh":
        sums["recall_nh_micro_num"] += hits
        sums["recall_nh_micro_den"] += len(target)


def eval_model_by_item_strata(
    model,
    label,
    checkpoint_path,
    dataset_name,
    seed,
    split_name,
    loader,
    uid_field,
    iid_field,
    target_sets,
    exclude_items,
    strata,
    item_counts,
    batch_size,
    device,
    topks,
    progress_every,
):
    started = time.perf_counter()
    max_k = max(topks)
    eval_users = sorted(target_sets.keys())
    metric_sums = new_metric_sums(topks)
    torch_device = torch.device(device)

    with torch.no_grad():
        total_batches = (len(eval_users) + batch_size - 1) // batch_size
        for batch_idx, offset in enumerate(range(0, len(eval_users), batch_size), start=1):
            batch_users = eval_users[offset : offset + batch_size]
            interaction = Interaction(
                {uid_field: torch.tensor(batch_users, dtype=torch.long, device=torch_device)}
            )
            scores = model.full_sort_predict(interaction).view(len(batch_users), -1)
            scores[:, 0] = -torch.inf
            for row_idx, user in enumerate(batch_users):
                excluded = [item for item in exclude_items[user] if item < scores.shape[1]]
                if excluded:
                    scores[row_idx, torch.tensor(excluded, dtype=torch.long, device=torch_device)] = -torch.inf
            top_items = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()

            for row_idx, user in enumerate(batch_users):
                relevant = target_sets[user]
                if not relevant:
                    continue
                recommended = [int(item) for item in top_items[row_idx]]
                recommended_set_by_topk = {
                    topk: set(recommended[:topk])
                    for topk in topks
                }
                for topk in topks:
                    sums = metric_sums[topk]
                    recommended_at_k = recommended[:topk]
                    recommended_set = recommended_set_by_topk[topk]
                    hits = len(recommended_set & relevant)
                    ideal_len = min(len(relevant), topk)
                    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_len))

                    sums["users"] += 1
                    sums["recall"] += hits / len(relevant)
                    sums["recall_micro_num"] += hits
                    sums["recall_micro_den"] += len(relevant)
                    sums["ndcg"] += dcg_at_k(recommended, relevant, topk) / idcg if idcg > 0 else 0.0
                    sums["non_head_recommended"] += len(recommended_set & strata["non_head"])
                    sums["recommended"] += len(recommended_at_k)

                    update_segment_metrics(sums, topk, recommended, relevant, strata["head"], "head")
                    update_segment_metrics(sums, topk, recommended, relevant, strata["torso"], "torso")
                    update_segment_metrics(sums, topk, recommended, relevant, strata["tail"], "tail")
                    update_segment_metrics(sums, topk, recommended, relevant, strata["non_head"], "nh")

            if progress_every and batch_idx % progress_every == 0:
                print(
                    f"{label} {dataset_name} seed={seed} {split_name}: "
                    f"batch {batch_idx}/{total_batches}",
                    flush=True,
                )

    rows = []
    elapsed = time.perf_counter() - started
    for topk in topks:
        sums = metric_sums[topk]

        def avg(metric, denom):
            count = sums[denom]
            return sums[metric] / count if count else 0.0

        rows.append(
            {
                "Dataset": dataset_name,
                "Model": label,
                "Seed": seed,
                "Split": split_name,
                "TopK": topk,
                "Users": sums["users"],
                "Train Item Count": item_counts["train_item_count"],
                "Head Item Count": item_counts["head_item_count"],
                "Torso Item Count": item_counts["torso_item_count"],
                "Tail Item Count": len(strata["tail"]),
                "Cold Item Count": len(strata["cold"]),
                "Checkpoint": checkpoint_path,
                "Elapsed Sec": round(elapsed, 2),
                "recall": sums["recall"] / sums["users"] if sums["users"] else 0.0,
                "recall_micro": (
                    sums["recall_micro_num"] / sums["recall_micro_den"]
                    if sums["recall_micro_den"]
                    else 0.0
                ),
                "ndcg": sums["ndcg"] / sums["users"] if sums["users"] else 0.0,
                "recall_head": avg("recall_head", "recall_head_users"),
                "ndcg_head": avg("ndcg_head", "ndcg_head_users"),
                "recall_torso": avg("recall_torso", "recall_torso_users"),
                "ndcg_torso": avg("ndcg_torso", "ndcg_torso_users"),
                "recall_tail": avg("recall_tail", "recall_tail_users"),
                "ndcg_tail": avg("ndcg_tail", "ndcg_tail_users"),
                "recall_nh": avg("recall_nh", "recall_nh_users"),
                "recall_nh_micro": (
                    sums["recall_nh_micro_num"] / sums["recall_nh_micro_den"]
                    if sums["recall_nh_micro_den"]
                    else 0.0
                ),
                "ndcg_nh": avg("ndcg_nh", "ndcg_nh_users"),
                "non_head_share": (
                    sums["non_head_recommended"] / sums["recommended"]
                    if sums["recommended"]
                    else 0.0
                ),
            }
        )
    return rows


def main():
    setup_utf8_stdio()
    args = parse_args()
    setup(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")

    spec = DATASET_SPECS[args.dataset]
    if args.output_prefix is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_prefix = Path("train_logs") / f"reduced_item_strata_{timestamp}"
    else:
        output_prefix = Path(args.output_prefix)

    print(f"Preparing data for {spec['display_name']} seed={args.seed}", flush=True)
    ref_config, _, train_data, valid_data, test_data = prepare_reference_data(spec, args)
    uid_field = ref_config["USER_ID_FIELD"]
    iid_field = ref_config["ITEM_ID_FIELD"]
    train_items = build_user_items(train_data, uid_field, iid_field)
    valid_targets = build_user_items(valid_data, uid_field, iid_field)
    test_targets = build_user_items(test_data, uid_field, iid_field)
    split_loaders = {"valid": valid_data, "test": test_data}
    split_targets = {"valid": valid_targets, "test": test_targets}
    exclude_by_split = {
        "valid": train_items,
        "test": union_item_sets(train_items, valid_targets),
    }
    item_counts = build_item_strata(train_data, split_loaders, uid_field, iid_field)

    checkpoint_specs = []
    if args.lightgcn_checkpoint:
        checkpoint_specs.append(("LightGCN", "LightGCN", args.lightgcn_checkpoint))
    if args.asym_checkpoint:
        checkpoint_specs.append(("AsymLightGCN", "AsymLightGCN", args.asym_checkpoint))
    if args.ncl_checkpoint:
        checkpoint_specs.append(("NCL", "NCL", args.ncl_checkpoint))

    rows = []
    for kind, label, checkpoint_path in checkpoint_specs:
        print(f"\nLoading {label}: {checkpoint_path}", flush=True)
        _, model = load_model(kind, label, checkpoint_path, spec, train_data.dataset, args)
        eval_models = [(label, model, None)]
        if kind == "AsymLightGCN":
            if args.asym_gammas is None:
                gammas = [0.25]
                use_gamma_labels = False
            else:
                gammas = list(args.asym_gammas)
                use_gamma_labels = True
            if args.include_asym_gamma0:
                gammas.append(0.0)
            gammas = unique_gammas(gammas)
            eval_models = []
            for gamma in gammas:
                if use_gamma_labels:
                    eval_label = f"AsymLightGCN (Gamma={format_gamma(gamma)} eval-only)"
                elif abs(gamma - 0.0) < 1e-12:
                    eval_label = "AsymLightGCN (Gamma=0 eval-only)"
                else:
                    eval_label = label
                eval_models.append((eval_label, model, gamma))

        for eval_label, eval_model, asym_gamma in eval_models:
            if asym_gamma is not None and hasattr(eval_model, "magnitude_gamma"):
                eval_model.magnitude_gamma = asym_gamma
            for split_name in args.splits:
                if not args.load_other_parameter:
                    eval_model.restore_user_e = None
                    eval_model.restore_item_e = None
                split_rows = eval_model_by_item_strata(
                    model=eval_model,
                    label=eval_label,
                    checkpoint_path=checkpoint_path,
                    dataset_name=spec["dataset_name"],
                    seed=args.seed,
                    split_name=split_name,
                    loader=split_loaders[split_name],
                    uid_field=uid_field,
                    iid_field=iid_field,
                    target_sets=split_targets[split_name],
                    exclude_items=exclude_by_split[split_name],
                    strata=item_counts["split_strata"][split_name],
                    item_counts=item_counts,
                    batch_size=args.batch_size,
                    device=args.device,
                    topks=sorted(set(args.topks)),
                    progress_every=args.progress_every,
                )
                rows.extend(split_rows)
                summary = ", ".join(
                    f"R@{row['TopK']}={row['recall']:.4f}, N@{row['TopK']}={row['ndcg']:.4f}, "
                    f"TailR@{row['TopK']}={row['recall_tail']:.4f}, NHShare@{row['TopK']}={row['non_head_share']:.4f}"
                    for row in split_rows
                )
                print(f"{eval_label} {split_name}: {summary}", flush=True)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved item-strata metrics to {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
