"""Short eval-only demo for the defense video.

The script loads a saved AsymLightGCN checkpoint and evaluates it on the
Amazon min5 test split. It does not train or update the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from asym_model import AsymLightGCN, CustomTrainer, seed_everything
from final_full_graph_config import DATASET_SPECS, build_base_config


DEFAULT_CHECKPOINT = (
    "saved/final_reduced/clean_amazon_reduced_min5/"
    "LightGCN-May-08-2026_02-37-43.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval-only AsymLightGCN demo from a saved checkpoint."
    )
    parser.add_argument("--dataset", default="amazon_reduced_min5")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output",
        default="train_logs/demo_eval_only_asym_amazon_min5.json",
        help="Where to save the compact metric report.",
    )
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    warnings.filterwarnings("ignore")
    sys.argv = [sys.argv[0]]
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    spec = DATASET_SPECS[args.dataset]
    seed_everything(args.seed)

    print("=" * 72)
    print("EVAL-ONLY DEMO: AsymLightGCN on Amazon min5")
    print("=" * 72)
    print(f"Dataset:    {spec['display_name']}")
    print(f"Seed:       {args.seed}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Mode:       test split, full ranking, no training")
    print()

    started_at = time.perf_counter()

    print("[1/4] Preparing configuration and data...")
    config_dict = build_base_config(
        spec,
        seed=args.seed,
        epochs=90,
        train_batch_size=8192,
        eval_batch_size=8192,
        eval_step=15,
        stopping_step=2,
    )
    config_dict.update(
        {
            "model": "LightGCN",
            "dataset": spec["dataset_name"],
            "use_gpu": args.device == "cuda",
            "gpu_id": "" if args.device == "cpu" else 0,
            "magnitude_calibration_gamma": 0.25,
        }
    )

    config = Config(
        model="LightGCN",
        dataset=spec["dataset_name"],
        config_file_list=["config.yaml"],
        config_dict=config_dict,
    )
    dataset = create_dataset(config)
    train_data, _, test_data = data_preparation(config, dataset)

    print("[2/4] Building AsymLightGCN and loading semantic components...")
    model = AsymLightGCN(config, train_data.dataset).to(config["device"])
    trainer = CustomTrainer(config, model)

    print("[3/4] Running checkpoint evaluation...")
    result = trainer.evaluate(test_data, model_file=str(checkpoint_path), show_progress=False)

    elapsed = time.perf_counter() - started_at
    report = {
        "dataset": spec["dataset_name"],
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "mode": "eval-only",
        "elapsed_sec": round(elapsed, 3),
        "metrics": {key: float(value) for key, value in result.items()},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/4] Evaluation complete.")
    print()
    print("Metrics")
    print("-" * 72)
    print(f"Recall@10: {result['recall@10']:.4f}  ({pct(result['recall@10'])})")
    print(f"Recall@50: {result['recall@50']:.4f}  ({pct(result['recall@50'])})")
    print(f"NDCG@10:   {result['ndcg@10']:.4f}  ({pct(result['ndcg@10'])})")
    print(f"NDCG@50:   {result['ndcg@50']:.4f}  ({pct(result['ndcg@50'])})")
    print("-" * 72)
    print(f"Saved compact report: {output_path}")
    print(f"Elapsed: {elapsed:.2f} sec")


if __name__ == "__main__":
    main()
