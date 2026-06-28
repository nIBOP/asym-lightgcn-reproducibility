import argparse
import json
import math
import sys
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
from recbole.utils import init_seed

from final_full_graph_config import DATASET_SPECS, build_base_config


DEFAULT_DATASETS = ["amazon_reduced"]
DEGREE_BUCKETS = ("train_degree=1", "train_degree=2", "train_degree>=3")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose reduced-dataset split artifacts and measure a no-training "
            "semantic-only baseline."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=[
            "movies_reduced",
            "amazon_reduced",
            "amazon_reduced_min3",
            "amazon_reduced_min4",
            "amazon_reduced_min5",
            "amazon_reduced_min6",
            "amazon_reduced_min8",
            "amazon_reduced_min10",
            "yelp_reduced",
            "yelp_reduced_min3",
            "yelp_reduced_min5",
        ],
        help="Reduced datasets to analyze.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--eval-step", type=int, default=15)
    parser.add_argument("--stopping-step", type=int, default=2)
    parser.add_argument(
        "--split-mode",
        choices=["rs", "loo"],
        default="rs",
        help=(
            "Evaluation split protocol. `rs` keeps the default per-user "
            "random 80/10/10 split; `loo` uses RecBole leave-one-out "
            "valid_and_test with random ordering."
        ),
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Run exact semantic-only full-sort metrics.",
    )
    parser.add_argument(
        "--semantic-splits",
        nargs="+",
        default=["test"],
        choices=["valid", "test"],
        help="Evaluation splits for semantic-only baseline.",
    )
    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=512,
        help="Number of users per semantic-only scoring batch.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used only for semantic-only matrix scoring.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Defaults to train_logs/reduced_semantic_diagnostics_<timestamp>.",
    )
    return parser.parse_args()


def setup_recbole(seed):
    init_seed(seed, reproducibility=True)
    # Avoid RecBole consuming arguments intended for this diagnostic script.
    sys.argv = [sys.argv[0]]


def load_data(spec, args):
    config_dict = build_base_config(
        spec,
        seed=args.seed,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_step=args.eval_step,
        stopping_step=args.stopping_step,
    )
    if args.split_mode == "loo":
        config_dict["eval_args"] = {
            "split": {"LS": "valid_and_test"},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        }
    config_dict["use_gpu"] = False
    config_dict["gpu_id"] = ""
    config = Config(
        model="LightGCN",
        dataset=spec["dataset_name"],
        config_file_list=["config.yaml"],
        config_dict=config_dict,
    )
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return config, dataset, train_data, valid_data, test_data


def counter_from_tensor(tensor):
    return Counter(tensor.cpu().numpy().tolist())


def read_raw_user_degrees(inter_path):
    counts = Counter()
    with Path(inter_path).open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            if not line.strip():
                continue
            user = line.split("\t", 1)[0]
            counts[user] += 1
    return counts


def quantiles(values):
    if len(values) == 0:
        return {"p50": None, "p90": None, "p99": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def degree_bucket(train_degree):
    if train_degree == 1:
        return "train_degree=1"
    if train_degree == 2:
        return "train_degree=2"
    return "train_degree>=3"


def safe_bucket_name(bucket):
    return {
        "train_degree=1": "train_degree_1",
        "train_degree=2": "train_degree_2",
        "train_degree>=3": "train_degree_ge3",
    }[bucket]


def summarize_raw_degrees(dataset_key, raw_user_degrees):
    hist = Counter(raw_user_degrees.values())
    users = len(raw_user_degrees)
    interactions = sum(raw_user_degrees.values())
    le2 = sum(count for degree, count in hist.items() if degree <= 2)
    ge3 = users - le2
    ge10 = sum(count for degree, count in hist.items() if degree >= 10)
    qs = quantiles(list(raw_user_degrees.values()))
    return {
        "dataset": dataset_key,
        "raw_users": users,
        "raw_interactions": interactions,
        "raw_avg_per_user": interactions / users if users else 0.0,
        "raw_users_degree_1": hist.get(1, 0),
        "raw_users_degree_1_pct": hist.get(1, 0) / users if users else 0.0,
        "raw_users_degree_2": hist.get(2, 0),
        "raw_users_degree_2_pct": hist.get(2, 0) / users if users else 0.0,
        "raw_users_degree_le2": le2,
        "raw_users_degree_le2_pct": le2 / users if users else 0.0,
        "raw_users_degree_ge3": ge3,
        "raw_users_degree_ge3_pct": ge3 / users if users else 0.0,
        "raw_users_degree_ge10": ge10,
        "raw_users_degree_ge10_pct": ge10 / users if users else 0.0,
        "raw_degree_p50": qs["p50"],
        "raw_degree_p90": qs["p90"],
        "raw_degree_p99": qs["p99"],
    }


def summarize_split(dataset_key, split_name, loader, uid_field, iid_field, train_user_counts):
    users = loader.dataset.inter_feat[uid_field].cpu().numpy()
    items = loader.dataset.inter_feat[iid_field].cpu().numpy()
    user_counts = Counter(users.tolist())
    item_counts = Counter(items.tolist())
    train_degrees_by_user = [train_user_counts[user] for user in user_counts]
    train_degrees_by_inter = [train_user_counts[user] for user in users.tolist()]
    bucket_counts = Counter(degree_bucket(train_user_counts[user]) for user in users.tolist())
    user_bucket_counts = Counter(degree_bucket(train_user_counts[user]) for user in user_counts)
    user_q = quantiles(train_degrees_by_user)
    inter_q = quantiles(train_degrees_by_inter)
    return {
        "dataset": dataset_key,
        "split": split_name,
        "interactions": int(len(users)),
        "users": int(len(user_counts)),
        "items": int(len(item_counts)),
        "avg_interactions_per_user": len(users) / len(user_counts) if user_counts else 0.0,
        "target_interactions_train_degree_1": bucket_counts["train_degree=1"],
        "target_interactions_train_degree_2": bucket_counts["train_degree=2"],
        "target_interactions_train_degree_ge3": bucket_counts["train_degree>=3"],
        "target_users_train_degree_1": user_bucket_counts["train_degree=1"],
        "target_users_train_degree_2": user_bucket_counts["train_degree=2"],
        "target_users_train_degree_ge3": user_bucket_counts["train_degree>=3"],
        "eval_user_train_degree_p50": user_q["p50"],
        "eval_user_train_degree_p90": user_q["p90"],
        "eval_user_train_degree_p99": user_q["p99"],
        "target_interaction_train_degree_p50": inter_q["p50"],
        "target_interaction_train_degree_p90": inter_q["p90"],
        "target_interaction_train_degree_p99": inter_q["p99"],
    }


def load_item_clusters(spec, dataset, iid_field):
    df_map = pd.read_csv(spec["item_mapping_path"], sep="\t", dtype=str)
    cluster_by_token = dict(
        zip(df_map["item_id:token"].astype(str), df_map["cluster_id:token"].astype(int))
    )
    id_to_token = dataset.field2id_token[iid_field]
    cluster_by_iid = {}
    for iid in range(len(id_to_token)):
        token = str(id_to_token[iid])
        cluster = cluster_by_token.get(token)
        if cluster is not None:
            cluster_by_iid[iid] = cluster
    return cluster_by_iid


def build_user_item_and_cluster_sets(train_data, uid_field, iid_field, cluster_by_iid):
    train_items = defaultdict(set)
    train_clusters = defaultdict(set)
    users = train_data.dataset.inter_feat[uid_field].cpu().numpy().tolist()
    items = train_data.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    for user, item in zip(users, items):
        train_items[user].add(item)
        cluster = cluster_by_iid.get(item)
        if cluster is not None:
            train_clusters[user].add(cluster)
    return train_items, train_clusters


def summarize_cluster_overlap(
    dataset_key,
    split_name,
    loader,
    uid_field,
    iid_field,
    train_items,
    train_clusters,
    train_user_counts,
    train_item_counts,
    cluster_by_iid,
):
    totals = Counter()
    hits = Counter()
    cold_item_interactions = 0
    exact_train_item_repeats = 0
    mapped = 0
    users = loader.dataset.inter_feat[uid_field].cpu().numpy().tolist()
    items = loader.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    for user, item in zip(users, items):
        bucket = degree_bucket(train_user_counts[user])
        cluster = cluster_by_iid.get(item)
        if item not in train_item_counts:
            cold_item_interactions += 1
        if item in train_items[user]:
            exact_train_item_repeats += 1
        if cluster is None:
            continue
        mapped += 1
        totals["all"] += 1
        totals[bucket] += 1
        if cluster in train_clusters[user]:
            hits["all"] += 1
            hits[bucket] += 1
    row = {
        "dataset": dataset_key,
        "split": split_name,
        "target_interactions": len(items),
        "mapped_cluster_interactions": mapped,
        "cold_item_interactions": cold_item_interactions,
        "cold_item_interactions_pct": cold_item_interactions / len(items) if items else 0.0,
        "exact_train_item_repeats": exact_train_item_repeats,
        "same_cluster_all": hits["all"],
        "same_cluster_all_pct": hits["all"] / totals["all"] if totals["all"] else 0.0,
    }
    for bucket in DEGREE_BUCKETS:
        safe_name = safe_bucket_name(bucket)
        row[f"{safe_name}_targets"] = totals[bucket]
        row[f"{safe_name}_same_cluster"] = hits[bucket]
        row[f"{safe_name}_same_cluster_pct"] = hits[bucket] / totals[bucket] if totals[bucket] else 0.0
    return row


def load_semantic_item_matrix(spec, dataset, iid_field):
    raw_embeddings = torch.load(spec["semantic_embs_path"], map_location="cpu")
    first_embedding = next(iter(raw_embeddings.values()))
    first_tensor = torch.as_tensor(first_embedding, dtype=torch.float32)
    raw_dim = int(first_tensor.numel())
    token_to_id = dataset.field2token_id[iid_field]
    item_semantics = torch.zeros((dataset.item_num, raw_dim), dtype=torch.float32)
    mapped = 0
    for token, iid in token_to_id.items():
        if token == "[PAD]":
            continue
        embedding = raw_embeddings.get(str(token))
        if embedding is None:
            continue
        item_semantics[iid] = torch.as_tensor(embedding, dtype=torch.float32)
        mapped += 1
    return item_semantics, mapped


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


def profile_from_train_items(item_semantics, train_items_for_user):
    item_ids = [item for item in train_items_for_user if item != 0]
    if not item_ids:
        return None
    profile = item_semantics[item_ids].mean(dim=0)
    norm = torch.linalg.vector_norm(profile)
    if not torch.isfinite(norm) or norm <= 0:
        return None
    return profile / norm


def evaluate_semantic_only(
    dataset_key,
    split_name,
    loader,
    uid_field,
    iid_field,
    item_semantics,
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
    item_semantics_cpu = F.normalize(item_semantics, p=2, dim=1).cpu()
    item_semantics_device = item_semantics_cpu.to(device)
    item_semantics_t = item_semantics_device.transpose(0, 1).contiguous()
    metric_sums = {
        topk: {"recall": 0.0, "ndcg": 0.0, "users": 0}
        for topk in topks
    }
    bucket_metric_sums = {
        bucket: {topk: {"recall": 0.0, "ndcg": 0.0, "users": 0} for topk in topks}
        for bucket in DEGREE_BUCKETS
    }
    no_profile_users = 0

    for offset in range(0, len(eval_users), batch_size):
        batch_users = eval_users[offset : offset + batch_size]
        profiles = []
        kept_users = []
        for user in batch_users:
            profile = profile_from_train_items(item_semantics_cpu, train_items[user])
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
            exclude = [item for item in exclude_items[user] if item < scores.shape[1]]
            if exclude:
                scores[row_idx, torch.tensor(exclude, dtype=torch.long, device=device)] = -torch.inf
        top_items = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()

        for row_idx, user in enumerate(kept_users):
            relevant = target_sets[user]
            if not relevant:
                continue
            bucket = degree_bucket(train_user_counts[user])
            recommended = top_items[row_idx]
            for topk in topks:
                top_recommended = recommended[:topk]
                hits = [1 if int(item) in relevant else 0 for item in top_recommended]
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
                    "dataset": dataset_key,
                    "split": split_name,
                    "segment": segment,
                    "topk": topk,
                    "users": users,
                    "recall": sums["recall"] / users if users else 0.0,
                    "ndcg": sums["ndcg"] / users if users else 0.0,
                    "no_profile_users": no_profile_users,
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
    return rows


def write_outputs(output_prefix, tables):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(tables, handle, indent=2, ensure_ascii=False)

    for name, rows in tables.items():
        if not rows:
            continue
        csv_path = output_prefix.with_name(f"{output_prefix.name}_{name}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    args = parse_args()
    setup_recbole(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")

    if args.output_prefix is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_prefix = Path("train_logs") / f"reduced_semantic_diagnostics_{timestamp}"
    else:
        output_prefix = Path(args.output_prefix)

    tables = {
        "raw_degrees": [],
        "splits": [],
        "cluster_overlap": [],
        "semantic_only": [],
    }
    split_protocol = (
        "LOO(valid_and_test, group_by=user, order=RO)"
        if args.split_mode == "loo"
        else "RS([0.8,0.1,0.1], group_by=user, order=RO)"
    )

    for dataset_key in args.datasets:
        spec = DATASET_SPECS[dataset_key]
        print(f"\n=== {spec['display_name']} ===", flush=True)
        config, dataset, train_data, valid_data, test_data = load_data(spec, args)
        uid_field = config["USER_ID_FIELD"]
        iid_field = config["ITEM_ID_FIELD"]
        inter_path = Path(spec["dataset_dir"]) / f"{spec['dataset_name']}.inter"

        raw_user_degrees = read_raw_user_degrees(inter_path)
        raw_summary = summarize_raw_degrees(dataset_key, raw_user_degrees)
        raw_summary["split_protocol"] = split_protocol
        tables["raw_degrees"].append(raw_summary)
        print(
            "Raw degree: "
            f"users={raw_summary['raw_users']}, avg={raw_summary['raw_avg_per_user']:.3f}, "
            f"degree<=2={raw_summary['raw_users_degree_le2_pct']:.2%}",
            flush=True,
        )

        train_user_counts = counter_from_tensor(train_data.dataset.inter_feat[uid_field])
        train_item_counts = counter_from_tensor(train_data.dataset.inter_feat[iid_field])
        split_loaders = {"train": train_data, "valid": valid_data, "test": test_data}
        for split_name, loader in split_loaders.items():
            row = summarize_split(dataset_key, split_name, loader, uid_field, iid_field, train_user_counts)
            row["split_protocol"] = split_protocol
            tables["splits"].append(row)
            if split_name in {"valid", "test"}:
                print(
                    f"{split_name}: users={row['users']}, interactions={row['interactions']}, "
                    f"target train_degree=1={row['target_users_train_degree_1']}, "
                    f">=3={row['target_users_train_degree_ge3']}",
                    flush=True,
                )

        cluster_by_iid = load_item_clusters(spec, dataset, iid_field)
        train_items, train_clusters = build_user_item_and_cluster_sets(
            train_data, uid_field, iid_field, cluster_by_iid
        )
        for split_name, loader in {"valid": valid_data, "test": test_data}.items():
            row = summarize_cluster_overlap(
                dataset_key,
                split_name,
                loader,
                uid_field,
                iid_field,
                train_items,
                train_clusters,
                train_user_counts,
                train_item_counts,
                cluster_by_iid,
            )
            row["split_protocol"] = split_protocol
            tables["cluster_overlap"].append(row)
            print(
                f"{split_name}: same semantic cluster={row['same_cluster_all_pct']:.2%}, "
                f"degree1 same cluster={row['train_degree_1_same_cluster_pct']:.2%}",
                flush=True,
            )

        if args.semantic_only:
            item_semantics, mapped_semantics = load_semantic_item_matrix(spec, dataset, iid_field)
            valid_targets = build_targets(valid_data, uid_field, iid_field)
            exclude_by_split = {
                "valid": train_items,
                "test": union_item_sets(train_items, valid_targets),
            }
            print(
                f"Semantic-only: mapped {mapped_semantics}/{dataset.item_num - 1} item embeddings",
                flush=True,
            )
            for split_name in args.semantic_splits:
                loader = {"valid": valid_data, "test": test_data}[split_name]
                semantic_rows = evaluate_semantic_only(
                    dataset_key=dataset_key,
                    split_name=split_name,
                    loader=loader,
                    uid_field=uid_field,
                    iid_field=iid_field,
                    item_semantics=item_semantics,
                    train_items=train_items,
                    exclude_items=exclude_by_split[split_name],
                    train_user_counts=train_user_counts,
                    batch_size=args.semantic_batch_size,
                    device=args.device,
                )
                for row in semantic_rows:
                    row["split_protocol"] = split_protocol
                tables["semantic_only"].extend(semantic_rows)
                all_rows = [row for row in semantic_rows if row["segment"] == "all"]
                metrics_text = ", ".join(
                    f"R@{row['topk']}={row['recall']:.4f}, N@{row['topk']}={row['ndcg']:.4f}"
                    for row in all_rows
                )
                print(f"Semantic-only {split_name}: {metrics_text}", flush=True)

    write_outputs(output_prefix, tables)
    print(f"\nSaved diagnostics to {output_prefix}.json and companion CSV files", flush=True)


if __name__ == "__main__":
    main()
