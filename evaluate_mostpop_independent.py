import argparse
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from final_full_graph_config import ALL_DATASET_KEYS, DATASET_SPECS, build_base_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Independently evaluate MostPop from train item counts, without RecBole Pop model."
    )
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASET_KEYS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--topks", nargs="+", type=int, default=[10, 50])
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
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def make_config(spec, seed, args):
    config_dict = build_base_config(
        spec,
        seed=seed,
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
    config_dict["model"] = "LightGCN"
    config_dict["dataset"] = spec["dataset_name"]
    return Config(
        model="LightGCN",
        dataset=spec["dataset_name"],
        config_file_list=["config.yaml"],
        config_dict=config_dict,
    )


def build_user_items(loader, uid_field, iid_field):
    user_items = defaultdict(set)
    users = loader.dataset.inter_feat[uid_field].cpu().numpy().tolist()
    items = loader.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    for user, item in zip(users, items):
        user_items[int(user)].add(int(item))
    return user_items


def union_user_items(*maps):
    combined = defaultdict(set)
    for item_map in maps:
        for user, items in item_map.items():
            combined[user].update(items)
    return combined


def build_item_strata(train_items, eval_items):
    train_item_counts = pd.Series(train_items).value_counts()
    total_train_items = len(train_item_counts)
    head_cutoff = max(1, int(total_train_items * 0.20))
    torso_cutoff = min(max(head_cutoff + 1, int(total_train_items * 0.50)), total_train_items)

    head_items = set(int(item) for item in train_item_counts.index[:head_cutoff])
    torso_items = set(int(item) for item in train_item_counts.index[head_cutoff:torso_cutoff])
    tail_items_train = set(int(item) for item in train_item_counts.index[torso_cutoff:])
    train_item_set = set(int(item) for item in train_item_counts.index)
    cold_items = set(int(item) for item in eval_items) - train_item_set
    tail_items = tail_items_train | cold_items
    return {
        "train_item_count": total_train_items,
        "head_item_count": len(head_items),
        "torso_item_count": len(torso_items),
        "tail_item_count": len(tail_items),
        "cold_item_count": len(cold_items),
        "head": head_items,
        "torso": torso_items,
        "tail": tail_items,
        "non_head": torso_items | tail_items,
    }


def dcg_at_k(recommendations, relevant):
    return sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommendations)
        if item in relevant
    )


def new_metric_sums(topks):
    return {
        topk: {
            "recall": 0.0,
            "recall_micro_num": 0,
            "recall_micro_den": 0,
            "ndcg": 0.0,
            "users": 0,
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
        for topk in sorted(set(topks))
    }


def update_segment_metrics(sums, top_recs, relevant, segment_items, prefix):
    target = relevant & segment_items
    if not target:
        return
    rec_set = set(top_recs)
    hits = len(rec_set & target)
    topk = len(top_recs)
    ideal_hits = min(len(target), topk)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    ndcg = dcg_at_k(top_recs, target) / idcg if idcg > 0 else 0.0

    sums[f"recall_{prefix}"] += hits / len(target)
    sums[f"recall_{prefix}_users"] += 1
    sums[f"ndcg_{prefix}"] += ndcg
    sums[f"ndcg_{prefix}_users"] += 1
    if prefix == "nh":
        sums["recall_nh_micro_num"] += hits
        sums["recall_nh_micro_den"] += len(target)


def evaluate_dataset_seed(dataset_key, seed, args):
    init_seed(seed, reproducibility=True)
    spec = DATASET_SPECS[dataset_key]
    data_started = time.perf_counter()
    config = make_config(spec, seed, args)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    data_elapsed = time.perf_counter() - data_started
    eval_started = time.perf_counter()
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]

    train_items = train_data.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    popularity = [int(item) for item, _ in Counter(train_items).most_common()]
    max_k = max(args.topks)

    train_by_user = build_user_items(train_data, uid_field, iid_field)
    valid_by_user = build_user_items(valid_data, uid_field, iid_field)
    test_by_user = build_user_items(test_data, uid_field, iid_field)
    exclude_by_user = union_user_items(train_by_user, valid_by_user)
    test_items = test_data.dataset.inter_feat[iid_field].cpu().numpy().tolist()
    strata = build_item_strata(train_items, test_items)

    metric_sums = new_metric_sums(args.topks)

    for user, relevant in test_by_user.items():
        if not relevant:
            continue
        excluded = exclude_by_user.get(user, set())
        recs = []
        for item in popularity:
            if item in excluded:
                continue
            recs.append(item)
            if len(recs) >= max_k:
                break
        rec_set_by_k = {}
        for topk in sorted(set(args.topks)):
            top_recs = recs[:topk]
            rec_set = rec_set_by_k.setdefault(topk, set(top_recs))
            hits = len(rec_set & relevant)
            recall = hits / len(relevant)
            ideal_hits = min(len(relevant), topk)
            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
            ndcg = dcg_at_k(top_recs, relevant) / idcg if idcg > 0 else 0.0
            metric_sums[topk]["recall"] += recall
            metric_sums[topk]["recall_micro_num"] += hits
            metric_sums[topk]["recall_micro_den"] += len(relevant)
            metric_sums[topk]["ndcg"] += ndcg
            metric_sums[topk]["users"] += 1
            metric_sums[topk]["non_head_recommended"] += len(rec_set & strata["non_head"])
            metric_sums[topk]["recommended"] += len(top_recs)
            update_segment_metrics(metric_sums[topk], top_recs, relevant, strata["head"], "head")
            update_segment_metrics(metric_sums[topk], top_recs, relevant, strata["torso"], "torso")
            update_segment_metrics(metric_sums[topk], top_recs, relevant, strata["tail"], "tail")
            update_segment_metrics(metric_sums[topk], top_recs, relevant, strata["non_head"], "nh")

    eval_elapsed = time.perf_counter() - eval_started
    rows = []
    split_protocol = (
        "LOO(valid_and_test, group_by=user, order=RO)"
        if args.split_mode == "loo"
        else "RS([0.8,0.1,0.1], group_by=user, order=RO)"
    )
    for topk, sums in metric_sums.items():
        users = sums["users"]

        def avg(metric, denom):
            count = sums[denom]
            return sums[metric] / count if count else 0.0

        rows.append(
            {
                "Dataset": spec["dataset_name"],
                "Display Dataset": spec["display_name"],
                "Model": "MostPopIndependent",
                "Seed": seed,
                "Split": "test",
                "Split Protocol": split_protocol,
                "TopK": topk,
                "Users": users,
                "Train Interactions": len(train_items),
                "Data Prep Time (sec)": round(data_elapsed, 2),
                "Train Time (sec)": 0.0,
                "Eval Time (sec)": round(eval_elapsed, 2),
                "Stratified Eval Time (sec)": 0.0,
                "Train Item Count": strata["train_item_count"],
                "Head Item Count": strata["head_item_count"],
                "Torso Item Count": strata["torso_item_count"],
                "Tail Item Count": strata["tail_item_count"],
                "Cold Item Count": strata["cold_item_count"],
                "recall": sums["recall"] / users if users else 0.0,
                "recall_micro": (
                    sums["recall_micro_num"] / sums["recall_micro_den"]
                    if sums["recall_micro_den"]
                    else 0.0
                ),
                "ndcg": sums["ndcg"] / users if users else 0.0,
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
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_prefix = Path(args.output_prefix or f"train_logs/mostpop_independent_{timestamp}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_key in args.datasets:
        for seed in args.seeds:
            print(f"Evaluating independent MostPop: {dataset_key} seed={seed}", flush=True)
            rows.extend(evaluate_dataset_seed(dataset_key, seed, args))

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Independent MostPop Check\n\n")
        handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        handle.write(frame.to_markdown(index=False))
        handle.write("\n")

    print(f"Saved independent MostPop check to {csv_path} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
