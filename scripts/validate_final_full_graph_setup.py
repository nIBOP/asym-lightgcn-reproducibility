import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import NCL

from asym_model import AsymLightGCN, seed_everything, setup_utf8_stdio
from final_full_graph_config import (
    ALL_DATASET_KEYS,
    DEFAULT_NCL_CONTRASTIVE_MIB_LIMIT,
    DEFAULT_DATASET_ORDER,
    DEFAULT_NCL_TRAIN_BATCH_SIZE,
    build_base_config,
    choose_ncl_train_batch_size,
    choose_ncl_num_clusters,
    estimate_ncl_contrastive_matrix_mib,
    resolve_dataset_specs,
)


_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

warnings.filterwarnings("ignore", message=".*A value is being set on a copy of a DataFrame or Series.*")
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-check the final full-graph setup without starting long training jobs."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASET_KEYS,
        default=DEFAULT_DATASET_ORDER,
        help="Datasets to validate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=120, help="Config epoch limit used in the smoke setup.")
    parser.add_argument("--train-batch-size", type=int, default=4096, help="Training batch size for the config.")
    parser.add_argument(
        "--ncl-train-batch-size",
        type=int,
        default=DEFAULT_NCL_TRAIN_BATCH_SIZE,
        help="NCL-specific training batch size cap used for the safety estimate.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4096, help="Evaluation batch size for the config.")
    parser.add_argument("--eval-step", type=int, default=10, help="Validation frequency for the config.")
    parser.add_argument("--stopping-step", type=int, default=4, help="Early stopping patience for the config.")
    parser.add_argument(
        "--ncl-max-contrastive-mib",
        type=float,
        default=DEFAULT_NCL_CONTRASTIVE_MIB_LIMIT,
        help="Safety limit used for the NCL contrastive score matrix estimate.",
    )
    return parser.parse_args()


def assert_paths_exist(spec):
    required_paths = {
        "centroids_path": spec["centroids_path"],
        "item_mapping_path": spec["item_mapping_path"],
        "semantic_embs_path": spec["semantic_embs_path"],
        "dataset_dir": spec["dataset_dir"],
    }
    missing = {name: path for name, path in required_paths.items() if not Path(path).exists()}
    if missing:
        raise FileNotFoundError(f"Missing required paths for {spec['dataset_name']}: {missing}")


def main():
    args = parse_args()
    sys.argv = [sys.argv[0]]
    seed_everything(args.seed)
    setup_utf8_stdio()

    specs = resolve_dataset_specs(args.datasets)
    report = []

    print("VALIDATING FINAL FULL-GRAPH SETUP")
    for spec in specs:
        print(f"\n=== {spec['display_name']} ===")
        assert_paths_exist(spec)
        print("Required paths: OK")

        config_dict = build_base_config(
            spec,
            seed=args.seed,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_step=args.eval_step,
            stopping_step=args.stopping_step,
        )

        light_config = Config(
            model="LightGCN",
            dataset=spec["dataset_name"],
            config_file_list=["config.yaml"],
            config_dict=config_dict,
        )
        dataset = create_dataset(light_config)
        train_data, valid_data, test_data = data_preparation(light_config, dataset)

        safe_ncl_clusters = choose_ncl_num_clusters(
            user_num=train_data.dataset.user_num,
            item_num=train_data.dataset.item_num,
        )
        ncl_loss_micro_batch_size = choose_ncl_train_batch_size(args.train_batch_size, args.ncl_train_batch_size)
        ncl_contrastive_matrix_mib = estimate_ncl_contrastive_matrix_mib(
            batch_size=ncl_loss_micro_batch_size,
            user_num=train_data.dataset.user_num,
            item_num=train_data.dataset.item_num,
        )
        print(
            f"Data prep: users={train_data.dataset.user_num}, items={train_data.dataset.item_num}, "
            f"train_interactions={train_data.dataset.inter_num}, train_batches={len(train_data)}, "
            f"valid_batches={len(valid_data)}, test_batches={len(test_data)}, "
            f"safe_ncl_clusters={safe_ncl_clusters}, ncl_train_batch_size={args.train_batch_size}, "
            f"ncl_loss_micro_batch_size={ncl_loss_micro_batch_size}, "
            f"ncl_contrastive_matrix_estimate={ncl_contrastive_matrix_mib:.1f} MiB"
        )
        if ncl_contrastive_matrix_mib > args.ncl_max_contrastive_mib:
            raise RuntimeError(
                f"NCL contrastive estimate is {ncl_contrastive_matrix_mib:.1f} MiB, above "
                f"--ncl-max-contrastive-mib={args.ncl_max_contrastive_mib:.1f}."
            )

        asym_model = AsymLightGCN(light_config, train_data.dataset).to(light_config["device"])
        print(
            f"AsymLightGCN init: OK | semantic_gating={asym_model.use_semantic_gating}, "
            f"magnitude_gamma={asym_model.magnitude_gamma}"
        )

        ncl_config = Config(
            model="NCL",
            dataset=spec["dataset_name"],
            config_file_list=["config.yaml"],
            config_dict={
                **config_dict,
                "ssl_temp": 0.1,
                "ssl_reg": 1e-7,
                "hyper_layers": 1,
                "alpha": 1.0,
                "proto_reg": 1e-7,
                "m_step": 1,
                "num_clusters": safe_ncl_clusters,
                "train_batch_size": args.train_batch_size,
                "loss_micro_batch_size": ncl_loss_micro_batch_size,
            },
        )
        ncl_model = NCL(ncl_config, train_data.dataset).to(ncl_config["device"])
        print(f"NCL init: OK | num_clusters={safe_ncl_clusters}")

        report.append(
            {
                "dataset": spec["dataset_name"],
                "display_name": spec["display_name"],
                "users": train_data.dataset.user_num,
                "items": train_data.dataset.item_num,
                "train_interactions": train_data.dataset.inter_num,
                "train_batches": len(train_data),
                "valid_batches": len(valid_data),
                "test_batches": len(test_data),
                "safe_ncl_clusters": safe_ncl_clusters,
                "ncl_train_batch_size": args.train_batch_size,
                "ncl_loss_micro_batch_size": ncl_loss_micro_batch_size,
                "ncl_contrastive_matrix_estimate_mib": round(ncl_contrastive_matrix_mib, 2),
                "asym_semantic_gating": bool(asym_model.use_semantic_gating),
            }
        )

        del asym_model
        del ncl_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs("train_logs", exist_ok=True)
    report_path = Path("train_logs") / "final_full_graph_setup_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nValidation report saved to {report_path}")


if __name__ == "__main__":
    main()
