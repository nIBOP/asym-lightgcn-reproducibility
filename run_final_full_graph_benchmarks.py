import argparse
import datetime
import gc
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN, NCL
from recbole.utils import get_trainer

from asym_model import (
    AsymLightGCN,
    CustomTrainer,
    OutputLogger,
    evaluate_stratified,
    format_seconds,
    log_with_timestamp,
    seed_everything,
    setup_utf8_stdio,
)
from final_full_graph_config import (
    ALL_DATASET_KEYS,
    DEFAULT_NCL_CONTRASTIVE_MIB_LIMIT,
    DEFAULT_DATASET_ORDER,
    DEFAULT_NCL_KMEANS_DEVICE,
    DEFAULT_NCL_TRAIN_BATCH_SIZE,
    FAISS_RECOMMENDED_MIN_POINTS_PER_CENTROID,
    build_base_config,
    choose_ncl_kmeans_device,
    choose_ncl_train_batch_size,
    choose_ncl_num_clusters,
    estimate_ncl_contrastive_matrix_mib,
    recommended_ncl_max_clusters,
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
        description="Run the final full-graph benchmark while preserving the current graph structure."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASET_KEYS,
        default=DEFAULT_DATASET_ORDER,
        help="Datasets to run. Default: movies amazon yelp",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=120, help="Maximum training epochs.")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4096,
        help="Training batch size for LightGCN and AsymLightGCN.",
    )
    parser.add_argument(
        "--ncl-train-batch-size",
        type=int,
        default=DEFAULT_NCL_TRAIN_BATCH_SIZE,
        help=(
            "NCL-specific loss micro-batch size. The outer NCL train_batch_size stays equal to "
            "--train-batch-size so epoch batch counts match other models, while NCL loss/backward "
            "is split internally for memory."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4096,
        help="Evaluation batch size used by RecBole.",
    )
    parser.add_argument(
        "--eval-step",
        type=int,
        default=10,
        help="Run validation once every N epochs to reduce expensive full-ranking evaluations.",
    )
    parser.add_argument(
        "--stopping-step",
        type=int,
        default=4,
        help="Early stopping patience in validation checks, not epochs.",
    )
    parser.add_argument(
        "--ncl-cluster-divisor",
        type=int,
        default=50,
        help="Dynamic NCL cluster count divisor.",
    )
    parser.add_argument(
        "--ncl-cluster-cap",
        type=int,
        default=500,
        help="Dynamic NCL cluster count cap.",
    )
    parser.add_argument(
        "--ncl-cluster-floor",
        type=int,
        default=50,
        help="Dynamic NCL cluster count floor.",
    )
    parser.add_argument(
        "--with-stratified",
        action="store_true",
        help="Also compute stratified head/torso/tail metrics after test evaluation.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["LightGCN", "NCL", "AsymLightGCN"],
        default=["LightGCN", "AsymLightGCN", "NCL"],
        help="Models to run for each selected dataset, in execution order.",
    )
    parser.add_argument(
        "--ncl-max-contrastive-mib",
        type=float,
        default=DEFAULT_NCL_CONTRASTIVE_MIB_LIMIT,
        help=(
            "Abort NCL before training if one contrastive score matrix is estimated above this MiB limit. "
            "Actual transient memory is higher than this estimate."
        ),
    )
    parser.add_argument(
        "--ncl-kmeans-device",
        choices=["auto", "cpu", "gpu"],
        default=DEFAULT_NCL_KMEANS_DEVICE,
        help=(
            "Where to run FAISS k-means for NCL E-steps. auto keeps GPU for well-sampled clustering "
            "and uses CPU when k is high relative to the item/user side."
        ),
    )
    parser.add_argument(
        "--ncl-kmeans-niter",
        type=int,
        default=25,
        help="FAISS k-means iterations per NCL E-step.",
    )
    parser.add_argument(
        "--ncl-min-points-per-centroid",
        type=int,
        default=FAISS_RECOMMENDED_MIN_POINTS_PER_CENTROID,
        help="Minimum points per centroid required before NCL uses a requested cluster count without an override.",
    )
    parser.add_argument(
        "--allow-unsafe-ncl-clusters",
        action="store_true",
        help="Allow NCL cluster counts that over-cluster the sparsest user/item side.",
    )
    return parser.parse_args()


def metric_snapshot(metrics, keys=None):
    keys = keys or ["recall@10", "ndcg@10", "recall@50", "ndcg@50"]
    parts = []
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            try:
                parts.append(f"{key}={float(value):.4f}")
            except (TypeError, ValueError):
                parts.append(f"{key}={value}")
    return ", ".join(parts)


def merge_stratified_metrics(result, stratified_result):
    if not stratified_result:
        return result
    for key, value in stratified_result.items():
        if key not in result:
            result[key] = value
    return result


def unique_in_order(values):
    seen = set()
    ordered_values = []
    for value in values:
        if value in seen:
            continue
        ordered_values.append(value)
        seen.add(value)
    return ordered_values


def cleanup_torch():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            print(f"[cleanup] Skipped torch.cuda.empty_cache after CUDA error: {exc}", file=sys.stderr)


class StableNCL(NCL):
    def _cfg(self, key, default=None):
        try:
            return self.config[key]
        except Exception:
            return default

    def get_norm_adj_mat(self):
        sparse_adj = super().get_norm_adj_mat()
        return sparse_adj.coalesce()

    def _sync_cuda(self):
        device = self.device if str(self.device).startswith("cuda") else None
        torch.cuda.synchronize(device)

    def run_kmeans(self, x):
        import faiss

        kmeans_device = str(self._cfg("ncl_kmeans_device", "gpu")).lower()
        use_gpu = kmeans_device == "gpu"
        niter = int(self._cfg("ncl_kmeans_niter", 25) or 25)
        seed = int(self._cfg("seed", 1234) or 1234)

        x = np.ascontiguousarray(x.astype("float32", copy=False))
        if use_gpu and torch.cuda.is_available():
            self._sync_cuda()
            torch.cuda.empty_cache()

        kmeans = faiss.Kmeans(
            d=self.latent_dim,
            k=self.k,
            gpu=use_gpu,
            niter=niter,
            seed=seed,
            min_points_per_centroid=1,
        )
        kmeans.train(x)
        _, cluster_ids = kmeans.index.search(x, 1)

        cluster_cents = np.ascontiguousarray(kmeans.centroids.astype("float32", copy=False))
        cluster_ids = np.ascontiguousarray(cluster_ids.reshape(-1).astype("int64", copy=False))
        del kmeans

        if use_gpu and torch.cuda.is_available():
            self._sync_cuda()
            torch.cuda.empty_cache()

        centroids = torch.from_numpy(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)
        node2cluster = torch.from_numpy(cluster_ids).to(self.device)
        return centroids, node2cluster


def get_compact_baseline_trainer_cls(base_trainer_cls):
    class LocalCompactBaselineTrainer(base_trainer_cls):
        def _cfg(self, key, default=None):
            try:
                return self.config[key]
            except Exception:
                return default

        def _normalize_losses(self, losses, epoch_idx):
            if isinstance(losses, tuple):
                if hasattr(self, "config") and "warm_up_step" in self.config and epoch_idx < self.config["warm_up_step"]:
                    losses = losses[:-1]
                return losses, sum(losses)
            return losses, losses

        def _micro_batch_loss(self, losses, chunk_size, batch_size):
            chunk_weight = float(chunk_size) / float(batch_size)
            if isinstance(losses, tuple):
                if self._cfg("ncl_loss_micro_batch_preserve_sum_terms", False) and len(losses) >= 2:
                    loss = losses[0] * chunk_weight + sum(losses[1:])
                else:
                    loss = sum(losses) * chunk_weight
                return loss
            return losses * chunk_weight

        def _finish_optimizer_step(self, scaler, enable_scaler):
            if getattr(self, "clip_grad_norm", None):
                from torch.nn.utils import clip_grad_norm_

                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            if enable_scaler and scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

        def _loss_scalar(self, loss):
            try:
                return float(loss.detach().item())
            except Exception:
                return None

        def _loss_context(self, epoch_idx, batch_idx, total_batches, chunk_start=None, chunk_end=None, batch_size=None):
            context = f"epoch={epoch_idx}, batch={batch_idx}/{total_batches}"
            if chunk_start is not None and chunk_end is not None and batch_size is not None:
                context += f", chunk={chunk_start}:{chunk_end}/{batch_size}"
            return context

        def _raise_backward_error(self, exc, context, loss_value):
            loss_text = "unknown" if loss_value is None else f"{loss_value:.6g}"
            raise RuntimeError(f"Backward failed at {context}; loss={loss_text}") from exc

        def _train_microbatched_interaction(
            self,
            interaction,
            epoch_idx,
            batch_idx,
            total_batches,
            loss_func,
            micro_batch_size,
            sync_loss,
            scaler,
            enable_scaler,
        ):
            batch_size = len(interaction)
            self.optimizer.zero_grad()
            batch_loss = 0.0

            for chunk_start in range(0, batch_size, micro_batch_size):
                chunk_end = min(chunk_start + micro_batch_size, batch_size)
                chunk = interaction[chunk_start:chunk_end]
                losses = loss_func(chunk)
                losses, unscaled_loss = self._normalize_losses(losses, epoch_idx)
                loss = self._micro_batch_loss(losses, chunk_end - chunk_start, batch_size)
                if chunk_start == 0:
                    loss = loss + sync_loss

                context = self._loss_context(epoch_idx, batch_idx, total_batches, chunk_start, chunk_end, batch_size)
                loss_value = self._loss_scalar(loss)
                self._check_nan(loss)
                try:
                    if enable_scaler and scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                except RuntimeError as exc:
                    self._raise_backward_error(exc, context, loss_value)
                if loss_value is None:
                    loss_value = self._loss_scalar(loss) or 0.0
                batch_loss += loss_value
                del chunk, losses, unscaled_loss, loss

            self._finish_optimizer_step(scaler=scaler, enable_scaler=enable_scaler)
            return batch_loss

        def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
            self.model.train()
            loss_func = loss_func or self.model.calculate_loss

            if "single_spec" in self.config and not self.config["single_spec"] and train_data.shuffle:
                train_data.sampler.set_epoch(epoch_idx)

            sync_loss = 0
            is_single_spec = self.config["single_spec"] if "single_spec" in self.config else True
            if getattr(self, "set_reduce_hook", None) and not is_single_spec:
                self.set_reduce_hook()
                sync_loss = getattr(self, "sync_grad_loss", lambda: 0)()

            scaler = getattr(self, "scaler", None)
            enable_scaler = getattr(self, "enable_scaler", False)
            enable_amp = getattr(self, "enable_amp", False)
            batch_interval = int(self._cfg("batch_log_interval", 0) or 0)
            log_epoch_summary = bool(self._cfg("log_epoch_summary", False))
            loss_micro_batch_size = int(self._cfg("loss_micro_batch_size", 0) or 0)
            total_loss = 0.0
            total_batches = len(train_data)

            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                if 0 < loss_micro_batch_size < len(interaction):
                    if enable_amp:
                        raise RuntimeError("loss_micro_batch_size is not supported with AMP in this benchmark trainer.")
                    loss_value = self._train_microbatched_interaction(
                        interaction=interaction,
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                        loss_func=loss_func,
                        micro_batch_size=loss_micro_batch_size,
                        sync_loss=sync_loss,
                        scaler=scaler,
                        enable_scaler=enable_scaler,
                    )
                    total_loss += loss_value
                    if batch_interval > 0 and batch_idx > 0 and batch_idx % batch_interval == 0:
                        print(f"   [Epoch {epoch_idx}] batch {batch_idx}/{total_batches} | loss={loss_value:.4f}")
                    continue

                self.optimizer.zero_grad()

                if enable_amp:
                    from torch.cuda import amp

                    with amp.autocast(enabled=True):
                        losses = loss_func(interaction)
                else:
                    losses = loss_func(interaction)

                losses, loss = self._normalize_losses(losses, epoch_idx)

                context = self._loss_context(epoch_idx, batch_idx, total_batches)
                loss_value = self._loss_scalar(loss)
                self._check_nan(loss)

                try:
                    if enable_scaler and scaler is not None:
                        scaler.scale(loss + sync_loss).backward()
                        self._finish_optimizer_step(scaler=scaler, enable_scaler=enable_scaler)
                    else:
                        (loss + sync_loss).backward()
                        self._finish_optimizer_step(scaler=scaler, enable_scaler=enable_scaler)
                except RuntimeError as exc:
                    self._raise_backward_error(exc, context, loss_value)

                if loss_value is None:
                    loss_value = self._loss_scalar(loss) or 0.0
                total_loss += loss_value

                if batch_interval > 0 and batch_idx > 0 and batch_idx % batch_interval == 0:
                    print(f"   [Epoch {epoch_idx}] batch {batch_idx}/{total_batches} | loss={loss_value:.4f}")

            if log_epoch_summary:
                print(f"[Epoch {epoch_idx}] finished | total_loss={total_loss:.4f}")
            return total_loss

        def _check_nan(self, loss):
            if not torch.isfinite(loss.detach()).all().item():
                loss_value = self._loss_scalar(loss)
                loss_text = "unknown" if loss_value is None else f"{loss_value:.6g}"
                raise ValueError(f"Training loss is not finite: {loss_text}")

        def _valid_epoch(self, valid_data, show_progress=False):
            valid_score, valid_result = super(LocalCompactBaselineTrainer, self)._valid_epoch(
                valid_data, show_progress=False
            )
            if self._cfg("log_validation_metrics", False):
                print(f"[Validation] metrics: {valid_result}")
            return valid_score, valid_result

    return LocalCompactBaselineTrainer


def prepare_data(model_name, dataset_name, config_dict):
    if "seed" in config_dict:
        seed_everything(int(config_dict["seed"]))
    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_file_list=["config.yaml"],
        config_dict=config_dict,
    )
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return config, train_data, valid_data, test_data


def maybe_run_stratified(trainer, test_data, train_data, enabled):
    if not enabled:
        return None, 0.0
    started_at = time.perf_counter()
    result = evaluate_stratified(trainer, test_data, train_data, k=10)
    elapsed = time.perf_counter() - started_at
    return result, elapsed


def run_baseline(model_name, spec, base_config, args):
    run_label = f"{model_name} / {spec['display_name']}"
    config_dict = dict(base_config)
    config_dict["model"] = model_name
    config_dict["dataset"] = spec["dataset_name"]
    ncl_loss_micro_batch_size = None
    if model_name == "NCL":
        ncl_loss_micro_batch_size = choose_ncl_train_batch_size(
            train_batch_size=args.train_batch_size,
            ncl_train_batch_size=args.ncl_train_batch_size,
        )
        config_dict["train_batch_size"] = args.train_batch_size
        config_dict["loss_micro_batch_size"] = ncl_loss_micro_batch_size
        config_dict["ncl_loss_micro_batch_preserve_sum_terms"] = True

    log_with_timestamp(f"[{run_label}] Preparing config and data")
    data_started_at = time.perf_counter()
    config, train_data, valid_data, test_data = prepare_data(model_name, spec["dataset_name"], config_dict)
    data_elapsed = time.perf_counter() - data_started_at
    actual_train_batch_size = int(config["train_batch_size"])

    chosen_clusters = None
    ncl_kmeans_device = None
    contrastive_matrix_mib = None
    recommended_max_clusters = None
    if model_name == "NCL":
        chosen_clusters = choose_ncl_num_clusters(
            user_num=train_data.dataset.user_num,
            item_num=train_data.dataset.item_num,
            divisor=args.ncl_cluster_divisor,
            cap=args.ncl_cluster_cap,
            floor=args.ncl_cluster_floor,
        )
        recommended_max_clusters = recommended_ncl_max_clusters(
            user_num=train_data.dataset.user_num,
            item_num=train_data.dataset.item_num,
            min_points_per_centroid=args.ncl_min_points_per_centroid,
        )
        if chosen_clusters > recommended_max_clusters and not args.allow_unsafe_ncl_clusters:
            log_with_timestamp(
                f"[{run_label}] Unsafe NCL cluster count: requested={chosen_clusters}, "
                f"recommended_max={recommended_max_clusters}, "
                f"min_points_per_centroid={args.ncl_min_points_per_centroid}"
            )
            raise RuntimeError(
                f"Refusing to train NCL with num_clusters={chosen_clusters}: the sparsest user/item side only "
                f"supports about {recommended_max_clusters} clusters at "
                f"{args.ncl_min_points_per_centroid} points/centroid. Lower --ncl-cluster-floor/--ncl-cluster-cap "
                f"or pass --allow-unsafe-ncl-clusters for an intentional stress run."
            )
        ncl_kmeans_device = choose_ncl_kmeans_device(
            num_clusters=chosen_clusters,
            user_num=train_data.dataset.user_num,
            item_num=train_data.dataset.item_num,
            requested=args.ncl_kmeans_device,
            min_points_per_centroid=args.ncl_min_points_per_centroid,
        )
        config["num_clusters"] = chosen_clusters
        config["ncl_kmeans_device"] = ncl_kmeans_device
        config["ncl_kmeans_niter"] = args.ncl_kmeans_niter
        contrastive_matrix_mib = estimate_ncl_contrastive_matrix_mib(
            batch_size=ncl_loss_micro_batch_size,
            user_num=train_data.dataset.user_num,
            item_num=train_data.dataset.item_num,
        )

    cluster_suffix = (
        f", ncl_num_clusters={chosen_clusters}, ncl_recommended_max_clusters={recommended_max_clusters}"
        if chosen_clusters is not None
        else ""
    )
    kmeans_suffix = f", ncl_kmeans_device={ncl_kmeans_device}" if ncl_kmeans_device is not None else ""
    ncl_memory_suffix = ""
    if contrastive_matrix_mib is not None:
        ncl_memory_suffix = (
            f", ncl_loss_micro_batch_size={ncl_loss_micro_batch_size}, "
            f"optimizer_updates_per_epoch={len(train_data)}, "
            f"ncl_contrastive_matrix_estimate={contrastive_matrix_mib:.1f} MiB"
        )
    log_with_timestamp(
        f"[{run_label}] Data ready in {format_seconds(data_elapsed)} | "
        f"users={train_data.dataset.user_num}, items={train_data.dataset.item_num}, "
        f"train_interactions={train_data.dataset.inter_num}, train_batches={len(train_data)}, "
        f"valid_batches={len(valid_data)}, test_batches={len(test_data)}"
        f"{cluster_suffix}{kmeans_suffix}{ncl_memory_suffix}"
    )
    if contrastive_matrix_mib is not None and contrastive_matrix_mib > args.ncl_max_contrastive_mib:
        raise RuntimeError(
            f"Refusing to train NCL with loss_micro_batch_size={ncl_loss_micro_batch_size}: one contrastive score matrix "
            f"is estimated at {contrastive_matrix_mib:.1f} MiB, above --ncl-max-contrastive-mib="
            f"{args.ncl_max_contrastive_mib:.1f}. Lower --ncl-train-batch-size."
        )

    if model_name == "LightGCN":
        model = LightGCN(config, train_data.dataset).to(config["device"])
    elif model_name == "NCL":
        model = StableNCL(config, train_data.dataset).to(config["device"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    base_trainer_cls = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer_cls = get_compact_baseline_trainer_cls(base_trainer_cls)
    trainer = trainer_cls(config, model)

    log_with_timestamp(f"[{run_label}] Training started")
    train_started_at = time.perf_counter()
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=False)
    train_elapsed = time.perf_counter() - train_started_at
    log_with_timestamp(
        f"[{run_label}] Training finished in {format_seconds(train_elapsed)} | "
        f"best_valid_score={best_valid_score:.4f} | "
        f"{metric_snapshot(best_valid_result, ['recall@10', 'ndcg@10'])}"
    )

    eval_started_at = time.perf_counter()
    test_result = trainer.evaluate(test_data, show_progress=False)
    eval_elapsed = time.perf_counter() - eval_started_at
    log_with_timestamp(
        f"[{run_label}] Test evaluation finished in {format_seconds(eval_elapsed)} | "
        f"{metric_snapshot(test_result)}"
    )

    stratified_result, stratified_elapsed = maybe_run_stratified(
        trainer=trainer,
        test_data=test_data,
        train_data=train_data,
        enabled=args.with_stratified,
    )
    if stratified_result is not None:
        log_with_timestamp(
            f"[{run_label}] Stratified evaluation finished in {format_seconds(stratified_elapsed)} | "
            f"{metric_snapshot(stratified_result, ['Head Recall', 'Torso Recall', 'Tail Recall'])}"
        )

    result = {
        "Dataset": spec["dataset_name"],
        "Display Dataset": spec["display_name"],
        "Model": model_name,
        "Seed": args.seed,
        "Eval Step": args.eval_step,
        "Stopping Step": args.stopping_step,
        "Epoch Limit": args.epochs,
        "Train Batch Size": actual_train_batch_size,
        "Eval Batch Size": args.eval_batch_size,
        "Data Prep Time (sec)": round(data_elapsed, 2),
        "Train Time (sec)": round(train_elapsed, 2),
        "Eval Time (sec)": round(eval_elapsed, 2),
        "Stratified Eval Time (sec)": round(stratified_elapsed, 2) if args.with_stratified else 0.0,
        **dict(test_result),
    }
    if chosen_clusters is not None:
        result["NCL Num Clusters"] = chosen_clusters
        result["NCL Recommended Max Clusters"] = recommended_max_clusters
        result["NCL KMeans Device"] = ncl_kmeans_device
        result["NCL Loss Micro Batch Size"] = ncl_loss_micro_batch_size
        result["NCL Optimizer Updates Per Epoch"] = len(train_data)
        result["NCL Contrastive Matrix Estimate (MiB)"] = round(contrastive_matrix_mib, 2)
    merge_stratified_metrics(result, stratified_result)
    return result


def run_asym(spec, base_config, args):
    config_dict = dict(base_config)
    config_dict["model"] = "LightGCN"
    config_dict["dataset"] = spec["dataset_name"]
    config_dict["magnitude_calibration_gamma"] = 0.25
    run_label = f"AsymLightGCN / {spec['display_name']}"

    log_with_timestamp(f"[{run_label}] Preparing config and data")
    data_started_at = time.perf_counter()
    config, train_data, valid_data, test_data = prepare_data("LightGCN", spec["dataset_name"], config_dict)
    data_elapsed = time.perf_counter() - data_started_at
    actual_train_batch_size = int(config["train_batch_size"])
    log_with_timestamp(
        f"[{run_label}] Data ready in {format_seconds(data_elapsed)} | "
        f"users={train_data.dataset.user_num}, items={train_data.dataset.item_num}, "
        f"train_interactions={train_data.dataset.inter_num}, train_batches={len(train_data)}, "
        f"valid_batches={len(valid_data)}, test_batches={len(test_data)}"
    )

    model = AsymLightGCN(config, train_data.dataset).to(config["device"])
    trainer = CustomTrainer(config, model)

    log_with_timestamp(f"[{run_label}] Training started")
    train_started_at = time.perf_counter()
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=False)
    train_elapsed = time.perf_counter() - train_started_at
    log_with_timestamp(
        f"[{run_label}] Training finished in {format_seconds(train_elapsed)} | "
        f"best_valid_score={best_valid_score:.4f} | "
        f"{metric_snapshot(best_valid_result, ['recall@10', 'ndcg@10'])}"
    )

    eval_started_at = time.perf_counter()
    test_result = trainer.evaluate(test_data, show_progress=False)
    eval_elapsed = time.perf_counter() - eval_started_at
    log_with_timestamp(
        f"[{run_label}] Test evaluation finished in {format_seconds(eval_elapsed)} | "
        f"{metric_snapshot(test_result)}"
    )

    stratified_result, stratified_elapsed = maybe_run_stratified(
        trainer=trainer,
        test_data=test_data,
        train_data=train_data,
        enabled=args.with_stratified,
    )
    if stratified_result is not None:
        log_with_timestamp(
            f"[{run_label}] Stratified evaluation finished in {format_seconds(stratified_elapsed)} | "
            f"{metric_snapshot(stratified_result, ['Head Recall', 'Torso Recall', 'Tail Recall'])}"
        )

    gamma025_row = {
        "Dataset": spec["dataset_name"],
        "Display Dataset": spec["display_name"],
        "Model": "AsymLightGCN",
        "Seed": args.seed,
        "Eval Step": args.eval_step,
        "Stopping Step": args.stopping_step,
        "Epoch Limit": args.epochs,
        "Train Batch Size": actual_train_batch_size,
        "Eval Batch Size": args.eval_batch_size,
        "Gamma": 0.25,
        "Weights Reused": False,
        "Data Prep Time (sec)": round(data_elapsed, 2),
        "Train Time (sec)": round(train_elapsed, 2),
        "Eval Time (sec)": round(eval_elapsed, 2),
        "Stratified Eval Time (sec)": round(stratified_elapsed, 2) if args.with_stratified else 0.0,
        **dict(test_result),
    }
    merge_stratified_metrics(gamma025_row, stratified_result)

    gamma0_label = f"AsymLightGCN (Gamma=0 eval-only) / {spec['display_name']}"
    model.magnitude_gamma = 0.0
    log_with_timestamp(f"[{gamma0_label}] Reusing trained weights and evaluating with gamma=0")

    gamma0_eval_started_at = time.perf_counter()
    test_result_gamma0 = trainer.evaluate(test_data, show_progress=False)
    gamma0_eval_elapsed = time.perf_counter() - gamma0_eval_started_at
    log_with_timestamp(
        f"[{gamma0_label}] Test evaluation finished in {format_seconds(gamma0_eval_elapsed)} | "
        f"{metric_snapshot(test_result_gamma0)}"
    )

    gamma0_stratified_result, gamma0_stratified_elapsed = maybe_run_stratified(
        trainer=trainer,
        test_data=test_data,
        train_data=train_data,
        enabled=args.with_stratified,
    )
    if gamma0_stratified_result is not None:
        log_with_timestamp(
            f"[{gamma0_label}] Stratified evaluation finished in {format_seconds(gamma0_stratified_elapsed)} | "
            f"{metric_snapshot(gamma0_stratified_result, ['Head Recall', 'Torso Recall', 'Tail Recall'])}"
        )

    gamma0_row = {
        "Dataset": spec["dataset_name"],
        "Display Dataset": spec["display_name"],
        "Model": "AsymLightGCN (Gamma=0 eval-only)",
        "Seed": args.seed,
        "Eval Step": args.eval_step,
        "Stopping Step": args.stopping_step,
        "Epoch Limit": args.epochs,
        "Train Batch Size": actual_train_batch_size,
        "Eval Batch Size": args.eval_batch_size,
        "Gamma": 0.0,
        "Weights Reused": True,
        "Data Prep Time (sec)": round(data_elapsed, 2),
        "Train Time (sec)": 0.0,
        "Eval Time (sec)": round(gamma0_eval_elapsed, 2),
        "Stratified Eval Time (sec)": round(gamma0_stratified_elapsed, 2) if args.with_stratified else 0.0,
        **dict(test_result_gamma0),
    }
    merge_stratified_metrics(gamma0_row, gamma0_stratified_result)

    return gamma025_row, gamma0_row


def save_partial_results(rows, path):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()
    args.models = unique_in_order(args.models)
    if args.train_batch_size <= 0:
        raise ValueError("--train-batch-size must be positive")
    if args.ncl_train_batch_size <= 0:
        raise ValueError("--ncl-train-batch-size must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.ncl_max_contrastive_mib <= 0:
        raise ValueError("--ncl-max-contrastive-mib must be positive")
    if args.ncl_kmeans_niter <= 0:
        raise ValueError("--ncl-kmeans-niter must be positive")
    if args.ncl_min_points_per_centroid <= 0:
        raise ValueError("--ncl-min-points-per-centroid must be positive")

    sys.argv = [sys.argv[0]]
    seed_everything(args.seed)
    setup_utf8_stdio()

    os.makedirs("train_logs", exist_ok=True)
    os.makedirs("saved/final_full_graph", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join("train_logs", f"final_full_graph_benchmarks_{timestamp}.txt")
    sys.stdout = OutputLogger(log_filename)
    sys.stderr = sys.stdout

    run_manifest = {
        "timestamp": timestamp,
        "datasets": args.datasets,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "ncl_loss_micro_batch_size": choose_ncl_train_batch_size(args.train_batch_size, args.ncl_train_batch_size),
        "ncl_max_contrastive_mib": args.ncl_max_contrastive_mib,
        "eval_batch_size": args.eval_batch_size,
        "eval_step": args.eval_step,
        "stopping_step": args.stopping_step,
        "ncl_cluster_divisor": args.ncl_cluster_divisor,
        "ncl_cluster_cap": args.ncl_cluster_cap,
        "ncl_cluster_floor": args.ncl_cluster_floor,
        "ncl_kmeans_device": args.ncl_kmeans_device,
        "ncl_kmeans_niter": args.ncl_kmeans_niter,
        "ncl_min_points_per_centroid": args.ncl_min_points_per_centroid,
        "allow_unsafe_ncl_clusters": args.allow_unsafe_ncl_clusters,
        "with_stratified": args.with_stratified,
        "models": args.models,
    }
    manifest_path = os.path.join("train_logs", f"final_full_graph_manifest_{timestamp}.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    dataset_specs = resolve_dataset_specs(args.datasets)

    print("FINAL FULL-GRAPH BENCHMARKS")
    print(f"Datasets: {', '.join(spec['dataset_name'] for spec in dataset_specs)}")
    print(f"Log file: {log_filename}")
    print(f"Manifest: {manifest_path}")
    print(
        "Acceleration protocol: "
        f"epochs={args.epochs}, eval_step={args.eval_step}, stopping_step={args.stopping_step}, "
        f"train_batch_size={args.train_batch_size}, "
        f"ncl_loss_micro_batch_size={choose_ncl_train_batch_size(args.train_batch_size, args.ncl_train_batch_size)}, "
        f"eval_batch_size={args.eval_batch_size}, "
        f"ncl_kmeans_device={args.ncl_kmeans_device}, ncl_kmeans_niter={args.ncl_kmeans_niter}, "
        f"ncl_min_points_per_centroid={args.ncl_min_points_per_centroid}"
    )
    print(f"Models: {', '.join(args.models)}")
    print("Gamma=0 is evaluated reusing the trained AsymLightGCN weights because gamma only affects inference.")

    all_rows = []
    partial_results_path = os.path.join("train_logs", f"final_full_graph_results_{timestamp}_partial.csv")
    for spec in dataset_specs:
        print(f"\n=== DATASET: {spec['display_name']} ===")
        os.makedirs(spec["checkpoint_dir"], exist_ok=True)
        base_config = build_base_config(
            spec,
            seed=args.seed,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_step=args.eval_step,
            stopping_step=args.stopping_step,
        )

        for model_index, model_name in enumerate(args.models, start=1):
            try:
                if model_name == "LightGCN":
                    print(f"\n--- {model_index}. Vanilla LightGCN ---")
                    light_row = run_baseline("LightGCN", spec, base_config, args)
                    all_rows.append(light_row)
                    save_partial_results(all_rows, partial_results_path)
                    log_with_timestamp(f"[LightGCN / {spec['display_name']}] Stored result | {metric_snapshot(light_row)}")

                elif model_name == "NCL":
                    print(f"\n--- {model_index}. NCL ---")
                    ncl_config = dict(base_config)
                    ncl_config.update(
                        {
                            "ssl_temp": 0.1,
                            "ssl_reg": 1e-7,
                            "hyper_layers": 1,
                            "alpha": 1.0,
                            "proto_reg": 1e-7,
                            "m_step": 1,
                        }
                    )
                    ncl_row = run_baseline("NCL", spec, ncl_config, args)
                    all_rows.append(ncl_row)
                    save_partial_results(all_rows, partial_results_path)
                    log_with_timestamp(f"[NCL / {spec['display_name']}] Stored result | {metric_snapshot(ncl_row)}")

                elif model_name == "AsymLightGCN":
                    print(f"\n--- {model_index}. AsymLightGCN + Gamma=0 eval-only ---")
                    asym_row, gamma0_row = run_asym(spec, base_config, args)
                    all_rows.append(asym_row)
                    all_rows.append(gamma0_row)
                    save_partial_results(all_rows, partial_results_path)
                    log_with_timestamp(f"[AsymLightGCN / {spec['display_name']}] Stored result | {metric_snapshot(asym_row)}")
                    log_with_timestamp(
                        f"[AsymLightGCN (Gamma=0 eval-only) / {spec['display_name']}] "
                        f"Stored result | {metric_snapshot(gamma0_row)}"
                    )
            finally:
                cleanup_torch()

    results_df = pd.DataFrame(all_rows)
    results_path = os.path.join("train_logs", f"final_full_graph_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)

    print("\nFINAL RESULTS")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
