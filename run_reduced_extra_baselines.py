import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from recbole.model.general_recommender import BPR
from recbole.utils import get_trainer, init_seed

from asym_model import SemanticGatedBPR
from final_full_graph_config import ALL_DATASET_KEYS, DATASET_SPECS, build_base_config, resolve_dataset_specs
from run_final_full_graph_benchmarks import (
    get_compact_baseline_trainer_cls,
    maybe_run_stratified,
    metric_snapshot,
    prepare_data,
    save_partial_results,
)


MODEL_CLASSES = {
    "BPR": BPR,
    "SemanticGatedBPR": SemanticGatedBPR,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run additional reduced-dataset RecBole trainable baselines. "
            "Use evaluate_mostpop_independent.py for the independent MostPop baseline."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASET_KEYS,
        default=["amazon_reduced_min5"],
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CLASSES), default=["BPR"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--eval-step", type=int, default=15)
    parser.add_argument("--stopping-step", type=int, default=2)
    parser.add_argument("--with-stratified", action="store_true")
    parser.add_argument("--checkpoint-root", default="saved/extra_reduced_baselines")
    return parser.parse_args()


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def cleanup_torch():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def canonical_model_name(model_name):
    if model_name == "SemanticGatedBPR":
        return "BPR"
    return model_name


def display_model_name(model_name):
    return model_name


def build_extra_config(spec, seed, model_name, args):
    epochs = 1 if canonical_model_name(model_name) == "Pop" else args.epochs
    eval_step = 1 if canonical_model_name(model_name) == "Pop" else args.eval_step
    stopping_step = 1 if canonical_model_name(model_name) == "Pop" else args.stopping_step
    config = build_base_config(
        spec,
        seed=seed,
        epochs=epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_step=eval_step,
        stopping_step=stopping_step,
    )
    config["checkpoint_dir"] = str(Path(args.checkpoint_root) / spec["dataset_name"] / display_model_name(model_name))
    config["model"] = canonical_model_name(model_name)
    config["dataset"] = spec["dataset_name"]
    return config


def run_one(spec, seed, model_name, args):
    canonical_name = canonical_model_name(model_name)
    row_model_name = display_model_name(model_name)
    config_dict = build_extra_config(spec, seed, model_name, args)
    init_seed(seed, reproducibility=True)
    os.makedirs(config_dict["checkpoint_dir"], exist_ok=True)

    run_label = f"{row_model_name} / {spec['display_name']} / seed {seed}"
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{run_label}] Preparing data", flush=True)
    data_started = time.perf_counter()
    config, train_data, valid_data, test_data = prepare_data(canonical_name, spec["dataset_name"], config_dict)
    data_elapsed = time.perf_counter() - data_started
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{run_label}] Data ready in {format_seconds(data_elapsed)} | "
        f"users={train_data.dataset.user_num}, items={train_data.dataset.item_num}, "
        f"train_interactions={train_data.dataset.inter_num}, train_batches={len(train_data)}, "
        f"valid_batches={len(valid_data)}, test_batches={len(test_data)}",
        flush=True,
    )

    model = MODEL_CLASSES[model_name](config, train_data.dataset).to(config["device"])
    base_trainer_cls = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer_cls = get_compact_baseline_trainer_cls(base_trainer_cls)
    trainer = trainer_cls(config, model)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{run_label}] Training started", flush=True)
    train_started = time.perf_counter()
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=False)
    train_elapsed = time.perf_counter() - train_started
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{run_label}] Training finished in {format_seconds(train_elapsed)} | "
        f"best_valid_score={best_valid_score:.4f} | {metric_snapshot(best_valid_result, ['recall@10', 'ndcg@10'])}",
        flush=True,
    )

    eval_started = time.perf_counter()
    test_result = trainer.evaluate(test_data, show_progress=False)
    eval_elapsed = time.perf_counter() - eval_started
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{run_label}] Test finished in {format_seconds(eval_elapsed)} | "
        f"{metric_snapshot(test_result)}",
        flush=True,
    )

    stratified_result, stratified_elapsed = maybe_run_stratified(
        trainer=trainer,
        test_data=test_data,
        train_data=train_data,
        enabled=args.with_stratified,
    )

    row = {
        "Dataset": spec["dataset_name"],
        "Display Dataset": spec["display_name"],
        "Model": row_model_name,
        "Seed": seed,
        "Eval Step": int(config["eval_step"]),
        "Stopping Step": int(config["stopping_step"]),
        "Epoch Limit": int(config["epochs"]),
        "Train Batch Size": int(config["train_batch_size"]),
        "Eval Batch Size": int(config["eval_batch_size"]),
        "Data Prep Time (sec)": round(data_elapsed, 2),
        "Train Time (sec)": round(train_elapsed, 2),
        "Eval Time (sec)": round(eval_elapsed, 2),
        "Stratified Eval Time (sec)": round(stratified_elapsed, 2) if args.with_stratified else 0.0,
        **dict(test_result),
    }
    if stratified_result:
        row.update(stratified_result)
    return row


def main():
    args = parse_args()
    sys.argv = [sys.argv[0]]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path("train_logs").mkdir(exist_ok=True)
    manifest_path = Path("train_logs") / f"extra_baselines_manifest_{timestamp}.json"
    partial_path = Path("train_logs") / f"extra_baselines_results_{timestamp}_partial.csv"
    results_path = Path("train_logs") / f"extra_baselines_results_{timestamp}.csv"

    specs = resolve_dataset_specs(args.datasets)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    rows = []
    for seed in args.seeds:
        for spec in specs:
            for model_name in args.models:
                try:
                    row = run_one(spec, seed, model_name, args)
                    rows.append(row)
                    save_partial_results(rows, partial_path)
                finally:
                    cleanup_torch()

    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
