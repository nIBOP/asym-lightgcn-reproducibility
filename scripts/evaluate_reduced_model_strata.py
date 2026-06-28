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
from final_full_graph_config import DATASET_SPECS, build_base_config


DEGREE_BUCKETS = ("train_degree=1", "train_degree=2", "train_degree>=3")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved reduced-dataset checkpoints by train-degree strata."
    )
    parser.add_argument(
        "--dataset",
        default="amazon_reduced",
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--lightgcn-checkpoint", default=None)
    parser.add_argument("--asym-checkpoint", default=None)
    parser.add_argument("--ncl-checkpoint", default=None)
    parser.add_argument("--include-asym-gamma0", action="store_true")
    parser.add_argument(
        "--load-other-parameter",
        action="store_true",
        help="Load RecBole checkpoint other_parameter cache such as restore_user_e/restore_item_e.",
    )
    parser.add_argument(
        "--legacy-asym-no-raw-gating",
        action="store_true",
        help=(
            "Force raw_semantic_embs=None for saved Asym checkpoints produced before the "
            "raw semantic gating buffer bug is fixed."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Defaults to train_logs/reduced_model_strata_<timestamp>.",
    )
    return parser.parse_args()


def setup(seed):
    init_seed(seed, reproducibility=True)
    sys.argv = [sys.argv[0]]


def degree_bucket(train_degree):
    if train_degree == 1:
        return "train_degree=1"
    if train_degree == 2:
        return "train_degree=2"
    return "train_degree>=3"


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


def build_sets(train_data, uid_field, iid_field):
    train_items = defaultdict(set)
    train_user_counts = Counter()
    users = train_data.dataset.inter_feat[uid_field].cpu().numpy().tolist()
    items = train_data.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    for user, item in zip(users, items):
        train_items[user].add(item)
        train_user_counts[user] += 1
    return train_items, train_user_counts


def build_targets(loader, uid_field, iid_field):
    targets = defaultdict(set)
    users = loader.dataset.inter_feat[uid_field].cpu().numpy().tolist()
    items = loader.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    for user, item in zip(users, items):
        targets[user].add(item)
    return targets


def union_item_sets(left, right):
    combined = defaultdict(set)
    for user, items in left.items():
        combined[user].update(items)
    for user, items in right.items():
        combined[user].update(items)
    return combined


def instantiate_model(kind, config, dataset, args):
    if kind == "LightGCN":
        return LightGCN(config, dataset)
    if kind == "AsymLightGCN":
        model = AsymLightGCN(config, dataset)
        if args.legacy_asym_no_raw_gating:
            model.raw_semantic_embs = None
        return model
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
                "num_clusters": 160,
                "m_step": 1,
            },
        )
    else:
        config = make_config(spec, args, "LightGCN")

    model = instantiate_model(kind, config, model_dataset, args)
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


def eval_model_by_strata(
    model,
    label,
    split_name,
    loader,
    uid_field,
    iid_field,
    train_items,
    exclude_items,
    train_user_counts,
    batch_size,
    device,
    topks=(10, 50),
):
    started = time.perf_counter()
    max_k = max(topks)
    target_sets = build_targets(loader, uid_field, iid_field)
    eval_users = sorted(target_sets.keys())
    metric_sums = {topk: {"recall": 0.0, "ndcg": 0.0, "users": 0} for topk in topks}
    bucket_metric_sums = {
        bucket: {topk: {"recall": 0.0, "ndcg": 0.0, "users": 0} for topk in topks}
        for bucket in DEGREE_BUCKETS
    }

    with torch.no_grad():
        for offset in range(0, len(eval_users), batch_size):
            batch_users = eval_users[offset : offset + batch_size]
            interaction = Interaction(
                {uid_field: torch.tensor(batch_users, dtype=torch.long, device=device)}
            )
            scores = model.full_sort_predict(interaction).view(len(batch_users), -1)
            scores[:, 0] = -torch.inf
            for row_idx, user in enumerate(batch_users):
                exclude = [item for item in exclude_items[user] if item < scores.shape[1]]
                if exclude:
                    scores[row_idx, torch.tensor(exclude, dtype=torch.long, device=device)] = -torch.inf
            top_items = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()

            for row_idx, user in enumerate(batch_users):
                relevant = target_sets[user]
                if not relevant:
                    continue
                bucket = degree_bucket(train_user_counts[user])
                recommended = top_items[row_idx]
                for topk in topks:
                    hits = [1 if int(item) in relevant else 0 for item in recommended[:topk]]
                    hit_count = sum(hits)
                    recall = hit_count / len(relevant)
                    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
                    ideal_len = min(len(relevant), topk)
                    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_len))
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    metric_sums[topk]["recall"] += recall
                    metric_sums[topk]["ndcg"] += ndcg
                    metric_sums[topk]["users"] += 1
                    bucket_metric_sums[bucket][topk]["recall"] += recall
                    bucket_metric_sums[bucket][topk]["ndcg"] += ndcg
                    bucket_metric_sums[bucket][topk]["users"] += 1

    rows = []
    for segment, sums_by_topk in [("all", metric_sums), *bucket_metric_sums.items()]:
        for topk in topks:
            sums = sums_by_topk[topk]
            users = sums["users"]
            rows.append(
                {
                    "model": label,
                    "split": split_name,
                    "segment": segment,
                    "topk": topk,
                    "users": users,
                    "recall": sums["recall"] / users if users else 0.0,
                    "ndcg": sums["ndcg"] / users if users else 0.0,
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
    return rows


def main():
    args = parse_args()
    setup(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")

    spec = DATASET_SPECS[args.dataset]
    if args.output_prefix is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_prefix = Path("train_logs") / f"reduced_model_strata_{timestamp}"
    else:
        output_prefix = Path(args.output_prefix)

    print(f"Preparing data for {spec['display_name']}", flush=True)
    ref_config, ref_dataset, train_data, valid_data, test_data = prepare_reference_data(spec, args)
    uid_field = ref_config["USER_ID_FIELD"]
    iid_field = ref_config["ITEM_ID_FIELD"]
    train_items, train_user_counts = build_sets(train_data, uid_field, iid_field)
    split_loaders = {"valid": valid_data, "test": test_data}
    valid_targets = build_targets(valid_data, uid_field, iid_field)
    exclude_by_split = {
        "valid": train_items,
        "test": union_item_sets(train_items, valid_targets),
    }

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
        for eval_label, eval_model in [(label, model)]:
            for split_name in args.splits:
                if not args.load_other_parameter:
                    eval_model.restore_user_e = None
                    eval_model.restore_item_e = None
                split_rows = eval_model_by_strata(
                    model=eval_model,
                    label=eval_label,
                    split_name=split_name,
                    loader=split_loaders[split_name],
                    uid_field=uid_field,
                    iid_field=iid_field,
                    train_items=train_items,
                    exclude_items=exclude_by_split[split_name],
                    train_user_counts=train_user_counts,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                rows.extend(split_rows)
                all_rows = [row for row in split_rows if row["segment"] == "all"]
                summary = ", ".join(
                    f"R@{row['topk']}={row['recall']:.4f}, N@{row['topk']}={row['ndcg']:.4f}"
                    for row in all_rows
                )
                print(f"{eval_label} {split_name}: {summary}", flush=True)

        if kind == "AsymLightGCN" and args.include_asym_gamma0:
            model.magnitude_gamma = 0.0
            for split_name in args.splits:
                if not args.load_other_parameter:
                    model.restore_user_e = None
                    model.restore_item_e = None
                split_rows = eval_model_by_strata(
                    model=model,
                    label="AsymLightGCN (Gamma=0 eval-only)",
                    split_name=split_name,
                    loader=split_loaders[split_name],
                    uid_field=uid_field,
                    iid_field=iid_field,
                    train_items=train_items,
                    exclude_items=exclude_by_split[split_name],
                    train_user_counts=train_user_counts,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                rows.extend(split_rows)
                all_rows = [row for row in split_rows if row["segment"] == "all"]
                summary = ", ".join(
                    f"R@{row['topk']}={row['recall']:.4f}, N@{row['topk']}={row['ndcg']:.4f}"
                    for row in all_rows
                )
                print(f"AsymLightGCN (Gamma=0 eval-only) {split_name}: {summary}", flush=True)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved strata metrics to {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
