import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from final_full_graph_config import DATASET_SPECS, build_base_config


DATE_TAG = "2026-05-08"
OUT_DIR = Path("train_logs")
PREFIX = OUT_DIR / f"paper_artifacts_{DATE_TAG}"

TARGET_DATASETS = [
    "amazon_reduced_min5",
    "movies_reduced",
    "yelp_reduced",
]

DATASET_LABELS = {
    "clean_amazon_reduced_min5": "Amazon min5",
    "clean_movies_reduced": "Movies",
    "clean_yelp_reduced": "Yelp",
}

MODEL_ORDER = [
    "MostPopIndependent",
    "Semantic-only",
    "Semantic-only diagnostic",
    "BPR",
    "LightGCN",
    "NCL",
    "NCL checkpoint eval, epoch 44",
    "AsymLightGCN",
    "AsymLightGCN, gamma=0 eval-only",
]

MODEL_LABELS = {
    "MostPopIndependent": "MostPop",
    "Semantic-only": "Semantic-only",
    "Semantic-only diagnostic": "Semantic-only",
    "BPR": "BPR-MF",
    "LightGCN": "LightGCN",
    "NCL": "NCL",
    "NCL checkpoint eval, epoch 44": "NCL checkpoint",
    "AsymLightGCN": "AsymLightGCN",
    "AsymLightGCN, gamma=0 eval-only": "AsymLightGCN gamma=0",
    "AsymLightGCN (Gamma=0 eval-only)": "AsymLightGCN gamma=0",
}


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def model_rank(model):
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def dataset_rank(dataset):
    order = ["clean_amazon_reduced_min5", "clean_movies_reduced", "clean_yelp_reduced"]
    try:
        return order.index(dataset)
    except ValueError:
        return len(order)


def fmt_float(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def display_value(row, metric):
    formatted = row.get(metric, "")
    if isinstance(formatted, str) and formatted:
        return formatted
    mean = row.get(f"{metric} mean", math.nan)
    std = row.get(f"{metric} std", math.nan)
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return fmt_float(mean)
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def add_main_labels(frame, source):
    frame = frame.copy()
    frame["Dataset Label"] = frame["Dataset"].map(DATASET_LABELS)
    frame["Model Label"] = frame["Model"].map(lambda value: MODEL_LABELS.get(value, value))
    frame["Source"] = str(source)
    return frame


def semantic_movies_yelp_main_rows(source):
    frame = safe_read_csv(source)
    frame = frame[(frame["split"].eq("test")) & (frame["segment"].eq("all"))].copy()
    dataset_map = {
        "movies_reduced": ("clean_movies_reduced", "Reduced Movies"),
        "yelp_reduced": ("clean_yelp_reduced", "Reduced Yelp"),
    }
    rows = []
    for raw_dataset, group in frame.groupby("dataset"):
        if raw_dataset not in dataset_map:
            continue
        by_topk = {int(row["topk"]): row for _, row in group.iterrows()}
        if 10 not in by_topk or 50 not in by_topk:
            continue
        dataset, display_dataset = dataset_map[raw_dataset]
        row10 = by_topk[10]
        row50 = by_topk[50]
        rows.append(
            {
                "Dataset": dataset,
                "Display Dataset": display_dataset,
                "Model": "Semantic-only",
                "Seeds": row10["seeds"],
                "Count": int(row10["count"]),
                "recall@10 mean": float(row10["recall_mean"]),
                "recall@10 std": float(row10["recall_std"]),
                "recall@10": row10["recall"],
                "ndcg@10 mean": float(row10["ndcg_mean"]),
                "ndcg@10 std": float(row10["ndcg_std"]),
                "ndcg@10": row10["ndcg"],
                "recall@50 mean": float(row50["recall_mean"]),
                "recall@50 std": float(row50["recall_std"]),
                "recall@50": row50["recall"],
                "ndcg@50 mean": float(row50["ndcg_mean"]),
                "ndcg@50 std": float(row50["ndcg_std"]),
                "ndcg@50": row50["ndcg"],
            }
        )
    return pd.DataFrame(rows)


def load_main_results():
    source = OUT_DIR / "paper_facing_reduced_results_corrected_2026-05-07_summary.csv"
    amazon_source = OUT_DIR / "amazon_min5_six_seed_summary_2026-05-08_summary.csv"
    semantic_source = OUT_DIR / "semantic_only_movies_yelp_3seed_summary_2026-05-08.csv"
    frame = safe_read_csv(source)
    frame = frame[~frame["Dataset"].eq("clean_amazon_reduced_min5")].copy()
    frame = frame[
        ~(
            frame["Dataset"].isin(["clean_movies_reduced", "clean_yelp_reduced"])
            & frame["Model"].isin(["Semantic-only", "Semantic-only diagnostic"])
        )
    ].copy()
    amazon = safe_read_csv(amazon_source)
    semantic = semantic_movies_yelp_main_rows(semantic_source)
    frame = pd.concat([frame, amazon, semantic], ignore_index=True, sort=False)
    keep = [
        "Dataset",
        "Display Dataset",
        "Model",
        "Seeds",
        "Count",
        "recall@10",
        "ndcg@10",
        "recall@50",
        "ndcg@50",
        "recall@10 mean",
        "ndcg@10 mean",
        "recall@50 mean",
        "ndcg@50 mean",
    ]
    frame = frame[keep].copy()
    frame = add_main_labels(frame, f"{source}; {amazon_source}; {semantic_source}")
    frame["_dataset_rank"] = frame["Dataset"].map(dataset_rank)
    frame["_model_rank"] = frame["Model"].map(model_rank)
    return frame.sort_values(["_dataset_rank", "_model_rank"]).drop(columns=["_dataset_rank", "_model_rank"])


def load_ablation_results():
    source = OUT_DIR / "asym_ablation_amazon_reduced_min5_complete_summary_2026-05-04.csv"
    frame = safe_read_csv(source).copy()
    keep = [
        "Ablation Key",
        "Ablation",
        "Use Semantic Gating",
        "Use Semantic Loss",
        "Use Centroid EMA",
        "Gating W Max",
        "recall@10 mean",
        "recall@10 std",
        "ndcg@10 mean",
        "ndcg@10 std",
        "recall@50 mean",
        "recall@50 std",
        "ndcg@50 mean",
        "ndcg@50 std",
        "recall_tail mean",
        "recall_tail std",
        "non_head_share mean",
        "non_head_share std",
    ]
    frame = frame[keep]
    frame["Dataset"] = "clean_amazon_reduced_min5"
    frame["Dataset Label"] = "Amazon min5"
    frame["Source"] = str(source)
    return frame


def normalize_strata_frame(frame, source):
    frame = frame.copy()
    rename_pairs = {
        "recall_mean": "recall mean",
        "recall_std": "recall std",
        "ndcg_mean": "ndcg mean",
        "ndcg_std": "ndcg std",
        "recall_head_mean": "recall_head mean",
        "recall_head_std": "recall_head std",
        "recall_tail_mean": "recall_tail mean",
        "recall_tail_std": "recall_tail std",
        "recall_nh_mean": "recall_nh mean",
        "recall_nh_std": "recall_nh std",
        "non_head_share_mean": "non_head_share mean",
        "non_head_share_std": "non_head_share std",
    }
    for src, dst in rename_pairs.items():
        if src in frame.columns and dst not in frame.columns:
            frame[dst] = frame[src]
    if "N" in frame.columns and "Count" not in frame.columns:
        frame["Count"] = frame["N"]
    if "Split" not in frame.columns:
        frame["Split"] = "test"
    if "Seeds" not in frame.columns:
        frame["Seeds"] = "42/43/44"
    if "recall mean" not in frame.columns:
        frame["recall mean"] = pd.to_numeric(frame.get("recall", math.nan), errors="coerce")
    if "ndcg mean" not in frame.columns:
        frame["ndcg mean"] = pd.to_numeric(frame.get("ndcg", math.nan), errors="coerce")
    for metric in ["recall_head", "recall_tail", "recall_nh", "non_head_share"]:
        if f"{metric} mean" not in frame.columns:
            frame[f"{metric} mean"] = pd.to_numeric(frame.get(metric, math.nan), errors="coerce")
    frame["Dataset Label"] = frame["Dataset"].map(DATASET_LABELS)
    frame["Model Label"] = frame["Model"].map(lambda value: MODEL_LABELS.get(value, value))
    frame["Source"] = str(source)
    return frame


def load_stratified_results():
    sources = [
        OUT_DIR / "amazon_min5_item_strata_six_seed_summary_2026-05-08.csv",
        OUT_DIR / "movies_yelp_core_stratified_3seed_summary_2026-05-06_summary.csv",
        OUT_DIR / "mostpop_independent_stratified_summary_2026-05-07_summary.csv",
    ]
    frames = [normalize_strata_frame(safe_read_csv(source), source) for source in sources]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = frame[frame["TopK"].astype(int).isin([10, 50])].copy()
    frame = frame.drop_duplicates(subset=["Dataset", "Model", "TopK"], keep="first")
    frame["_dataset_rank"] = frame["Dataset"].map(dataset_rank)
    frame["_model_rank"] = frame["Model"].map(model_rank)
    keep = [
        "Dataset",
        "Dataset Label",
        "Model",
        "Model Label",
        "Split",
        "TopK",
        "Seeds",
        "Count",
        "recall mean",
        "ndcg mean",
        "recall_head mean",
        "recall_tail mean",
        "recall_nh mean",
        "non_head_share mean",
        "Source",
    ]
    return frame.sort_values(["_dataset_rank", "TopK", "_model_rank"]).drop(columns=["_dataset_rank", "_model_rank"])[keep]


def fmt_seconds(mean, std):
    if pd.isna(std):
        return f"{float(mean):.1f}s"
    return f"{float(mean):.1f}s +/- {float(std):.1f}s"


def amazon_min5_runtime_overrides():
    selected_source = OUT_DIR / "amazon_min5_six_seed_summary_2026-05-08_selected_rows.csv"
    selected = safe_read_csv(selected_source)
    selected = selected[selected["Model"].isin(["BPR", "LightGCN", "NCL", "AsymLightGCN"])].copy()
    rows = []
    for _, selected_row in selected.iterrows():
        source_csv = Path(selected_row["Source CSV"])
        source_frame = safe_read_csv(source_csv)
        match = source_frame[
            source_frame["Dataset"].eq(selected_row["Dataset"])
            & source_frame["Model"].eq(selected_row["Model"])
            & (source_frame["Seed"].astype(int) == int(selected_row["Seed"]))
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        train_sec = float(row["Train Time (sec)"])
        eval_sec = float(row["Eval Time (sec)"])
        strat_sec = float(row.get("Stratified Eval Time (sec)", 0.0) or 0.0)
        rows.append(
            {
                "Dataset": selected_row["Dataset"],
                "Display Dataset": selected_row["Display Dataset"],
                "Model": selected_row["Model"],
                "Seed": int(selected_row["Seed"]),
                "Train Time (sec)": train_sec,
                "Eval Time (sec)": eval_sec,
                "Total Runtime (sec)": train_sec + eval_sec + strat_sec,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary_rows = []
    for (dataset, display_dataset, model), group in frame.groupby(["Dataset", "Display Dataset", "Model"]):
        seeds = "/".join(str(seed) for seed in sorted(group["Seed"].astype(int).unique()))
        train = group["Train Time (sec)"]
        eval_time = group["Eval Time (sec)"]
        total = group["Total Runtime (sec)"]
        summary_rows.append(
            {
                "Dataset": dataset,
                "Display Dataset": display_dataset,
                "Model": model,
                "Runtime Type": "train+eval",
                "Seeds": seeds,
                "Count": int(len(group)),
                "Train Time": fmt_seconds(train.mean(), train.std(ddof=1)),
                "Eval Time": fmt_seconds(eval_time.mean(), eval_time.std(ddof=1)),
                "Total Runtime": fmt_seconds(total.mean(), total.std(ddof=1)),
                "Train Time (sec) mean": float(train.mean()),
                "Eval Time (sec) mean": float(eval_time.mean()),
                "Total Runtime (sec) mean": float(total.mean()),
                "Source": str(selected_source),
            }
        )
    return pd.DataFrame(summary_rows)


def load_runtime_results():
    source = OUT_DIR / "runtime_summary_2026-05-07_summary.csv"
    frame = safe_read_csv(source).copy()
    overrides = amazon_min5_runtime_overrides()
    if not overrides.empty:
        frame = frame[
            ~(
                frame["Dataset"].eq("clean_amazon_reduced_min5")
                & frame["Model"].isin(overrides["Model"].unique())
                & frame["Runtime Type"].eq("train+eval")
            )
        ].copy()
        frame = pd.concat([frame, overrides], ignore_index=True, sort=False)
    frame["Dataset Label"] = frame["Dataset"].map(DATASET_LABELS)
    frame["Model Label"] = frame["Model"].map(lambda value: MODEL_LABELS.get(value, value))
    if "Source" not in frame.columns:
        frame["Source"] = str(source)
    else:
        frame["Source"] = frame["Source"].fillna(str(source))
    frame["_dataset_rank"] = frame["Dataset"].map(dataset_rank)
    frame["_model_rank"] = frame["Model"].map(model_rank)
    keep = [
        "Dataset",
        "Dataset Label",
        "Model",
        "Model Label",
        "Runtime Type",
        "Seeds",
        "Count",
        "Train Time",
        "Eval Time",
        "Total Runtime",
        "Train Time (sec) mean",
        "Eval Time (sec) mean",
        "Total Runtime (sec) mean",
        "Source",
    ]
    return frame.sort_values(["_dataset_rank", "_model_rank"]).drop(columns=["_dataset_rank", "_model_rank"])[keep]


def load_min_degree_sensitivity():
    source = OUT_DIR / "amazon_min_degree_sensitivity_summary_2026-05-08_summary.csv"
    frame = safe_read_csv(source).copy()
    frame["Source"] = str(source)
    return frame


def count_split(loader, uid_field, iid_field):
    users = loader.dataset.inter_feat[uid_field].cpu().numpy()
    items = loader.dataset.inter_feat[iid_field].cpu().numpy()
    return {
        "users": int(pd.Series(users).nunique()),
        "items": int(pd.Series(items).nunique()),
        "interactions": int(len(users)),
    }


def split_stats(spec):
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed

    init_seed(42, reproducibility=True)
    config_dict = build_base_config(
        spec,
        seed=42,
        epochs=90,
        train_batch_size=8192,
        eval_batch_size=8192,
        eval_step=15,
        stopping_step=2,
    )
    config_dict["model"] = "LightGCN"
    config_dict["dataset"] = spec["dataset_name"]
    config = Config(
        model="LightGCN",
        dataset=spec["dataset_name"],
        config_file_list=["config.yaml"],
        config_dict=config_dict,
    )
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    train = count_split(train_data, uid_field, iid_field)
    valid = count_split(valid_data, uid_field, iid_field)
    test = count_split(test_data, uid_field, iid_field)
    return {
        "Train Users (seed42)": train["users"],
        "Valid Users (seed42)": valid["users"],
        "Test Users (seed42)": test["users"],
        "Train Interactions (seed42)": train["interactions"],
        "Valid Interactions (seed42)": valid["interactions"],
        "Test Interactions (seed42)": test["interactions"],
    }


def load_dataset_stats():
    sys.argv = [sys.argv[0]]
    rows = []
    for key in TARGET_DATASETS:
        spec = DATASET_SPECS[key]
        inter_path = Path(spec["dataset_dir"]) / f"{spec['dataset_name']}.inter"
        frame = pd.read_csv(inter_path, sep="\t")
        user_col = "user_id:token"
        item_col = "item_id:token"
        user_degree = frame.groupby(user_col).size()
        row = {
            "Dataset": spec["dataset_name"],
            "Dataset Label": DATASET_LABELS[spec["dataset_name"]],
            "Users": int(frame[user_col].nunique()),
            "Items": int(frame[item_col].nunique()),
            "Interactions": int(len(frame)),
            "Avg Interactions/User": round(float(user_degree.mean()), 3),
            "Median Interactions/User": round(float(user_degree.median()), 3),
            "Users deg=1": int((user_degree == 1).sum()),
            "Users deg=2": int((user_degree == 2).sum()),
            "Users deg>=3": int((user_degree >= 3).sum()),
            "Share deg<=2": round(float((user_degree <= 2).mean()), 4),
            "Source": str(inter_path),
        }
        print(f"Preparing split stats for {spec['display_name']} seed=42", flush=True)
        row.update(split_stats(spec))
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure_data(main_results, stratified_results, runtime_results):
    main_fig = main_results[
        main_results["Model"].isin(["MostPopIndependent", "BPR", "LightGCN", "NCL", "AsymLightGCN"])
    ].copy()
    main_fig = main_fig[
        [
            "Dataset",
            "Dataset Label",
            "Model",
            "Model Label",
            "recall@10 mean",
            "ndcg@10 mean",
            "recall@50 mean",
            "ndcg@50 mean",
        ]
    ]

    niche_fig = stratified_results[
        (stratified_results["TopK"].astype(int) == 10)
        & stratified_results["Model"].isin(["MostPopIndependent", "LightGCN", "AsymLightGCN"])
    ].copy()
    niche_fig = niche_fig[
        [
            "Dataset",
            "Dataset Label",
            "Model",
            "Model Label",
            "recall_tail mean",
            "recall_nh mean",
            "non_head_share mean",
        ]
    ]

    runtime_fig = runtime_results[
        runtime_results["Runtime Type"].eq("train+eval")
        & runtime_results["Model"].isin(["BPR", "LightGCN", "NCL", "AsymLightGCN"])
    ].copy()
    runtime_fig = runtime_fig[
        [
            "Dataset",
            "Dataset Label",
            "Model",
            "Model Label",
            "Train Time (sec) mean",
            "Eval Time (sec) mean",
            "Total Runtime (sec) mean",
        ]
    ]
    return main_fig, niche_fig, runtime_fig


def maybe_write_pngs(main_fig, niche_fig, runtime_fig):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Matplotlib unavailable: {exc}"]

    messages = []
    try:
        pivot = main_fig.pivot(index="Dataset Label", columns="Model Label", values="recall@10 mean")
        pivot = pivot[[col for col in ["MostPop", "BPR-MF", "LightGCN", "NCL", "AsymLightGCN"] if col in pivot.columns]]
        ax = pivot.plot(kind="bar", figsize=(9, 4), rot=0)
        ax.set_ylabel("Recall@10")
        ax.set_xlabel("")
        ax.set_title("Main Recall@10")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        path = PREFIX.with_name(PREFIX.name + "_figure_main_recall10.png")
        plt.savefig(path, dpi=180)
        plt.close()
        messages.append(str(path))
    except Exception as exc:
        messages.append(f"main figure failed: {exc}")

    try:
        subset = niche_fig[niche_fig["Dataset"].isin(["clean_amazon_reduced_min5", "clean_yelp_reduced"])]
        pivot = subset.pivot(index="Dataset Label", columns="Model Label", values="non_head_share mean")
        pivot = pivot[[col for col in ["MostPop", "LightGCN", "AsymLightGCN"] if col in pivot.columns]]
        ax = pivot.plot(kind="bar", figsize=(7, 4), rot=0)
        ax.set_ylabel("Non-head share@10")
        ax.set_xlabel("")
        ax.set_title("Non-head Recommendation Share")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        path = PREFIX.with_name(PREFIX.name + "_figure_non_head_share10.png")
        plt.savefig(path, dpi=180)
        plt.close()
        messages.append(str(path))
    except Exception as exc:
        messages.append(f"niche figure failed: {exc}")

    try:
        subset = runtime_fig[runtime_fig["Dataset"].isin(["clean_movies_reduced", "clean_yelp_reduced"])]
        pivot = subset.pivot(index="Dataset Label", columns="Model Label", values="Total Runtime (sec) mean") / 60.0
        pivot = pivot[[col for col in ["BPR-MF", "LightGCN", "NCL", "AsymLightGCN"] if col in pivot.columns]]
        ax = pivot.plot(kind="bar", figsize=(8, 4), rot=0)
        ax.set_ylabel("Total runtime, minutes")
        ax.set_xlabel("")
        ax.set_title("Runtime on Large Reduced Datasets")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        path = PREFIX.with_name(PREFIX.name + "_figure_runtime_minutes.png")
        plt.savefig(path, dpi=180)
        plt.close()
        messages.append(str(path))
    except Exception as exc:
        messages.append(f"runtime figure failed: {exc}")
    return messages


def write_markdown(dataset_stats, main_results, ablations, stratified, runtime, min_degree, figure_messages):
    md_path = PREFIX.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Paper Artifacts Package\n\n")
        handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n")
        handle.write("This package consolidates the current paper-facing reduced-dataset tables. It excludes invalid RecBole `Pop` rows and uses `MostPopIndependent` instead.\n\n")

        handle.write("## Dataset Statistics\n\n")
        handle.write(dataset_stats.to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Main Results\n\n")
        cols = ["Dataset Label", "Model Label", "Seeds", "Count", "recall@10", "ndcg@10", "recall@50", "ndcg@50"]
        handle.write(main_results[cols].to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Amazon Min-Degree Sensitivity\n\n")
        min_degree_cols = ["Display Dataset", "Model", "Seeds", "Count", "recall@10", "ndcg@10", "recall@50", "ndcg@50"]
        handle.write(min_degree[min_degree_cols].to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Amazon Min5 Ablations\n\n")
        ablation_cols = [
            "Ablation Key",
            "Ablation",
            "Use Semantic Gating",
            "Use Semantic Loss",
            "Use Centroid EMA",
            "Gating W Max",
            "recall@10 mean",
            "ndcg@10 mean",
            "recall@50 mean",
            "ndcg@50 mean",
            "recall_tail mean",
            "non_head_share mean",
        ]
        handle.write(ablations[ablation_cols].to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Stratified K=10 Snapshot\n\n")
        strat_k10 = stratified[stratified["TopK"].astype(int) == 10].copy()
        strat_cols = [
            "Dataset Label",
            "Model Label",
            "Seeds",
            "recall mean",
            "ndcg mean",
            "recall_head mean",
            "recall_tail mean",
            "recall_nh mean",
            "non_head_share mean",
        ]
        handle.write(strat_k10[strat_cols].to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Runtime Summary\n\n")
        runtime_cols = [
            "Dataset Label",
            "Model Label",
            "Runtime Type",
            "Seeds",
            "Train Time",
            "Eval Time",
            "Total Runtime",
        ]
        handle.write(runtime[runtime_cols].to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Figure Artifacts\n\n")
        for message in figure_messages:
            handle.write(f"- `{message}`\n")
        handle.write("\n")

        handle.write("## Caveats\n\n")
        handle.write("- Yelp NCL is a validated checkpoint fallback at epoch 44, not a completed 90-epoch trainer run.\n")
        handle.write("- Movies NCL is a single completed seed-42 row from the older k=160 run.\n")
        handle.write("- Semantic-only is a no-training diagnostic baseline; Amazon min5 now has six seed, while Movies/Yelp have three seed.\n")
        handle.write("- Gamma=0 rows are eval-only reuse of Asym weights, not separate training runs.\n")
    return md_path


def main():
    OUT_DIR.mkdir(exist_ok=True)
    dataset_stats = load_dataset_stats()
    main_results = load_main_results()
    ablations = load_ablation_results()
    stratified = load_stratified_results()
    runtime = load_runtime_results()
    min_degree = load_min_degree_sensitivity()
    main_fig, niche_fig, runtime_fig = make_figure_data(main_results, stratified, runtime)
    figure_messages = maybe_write_pngs(main_fig, niche_fig, runtime_fig)

    outputs = {
        "dataset_stats": PREFIX.with_name(PREFIX.name + "_dataset_stats.csv"),
        "main_results": PREFIX.with_name(PREFIX.name + "_main_results.csv"),
        "ablation_results": PREFIX.with_name(PREFIX.name + "_ablation_results.csv"),
        "stratified_results": PREFIX.with_name(PREFIX.name + "_stratified_results.csv"),
        "runtime_results": PREFIX.with_name(PREFIX.name + "_runtime_results.csv"),
        "min_degree_sensitivity": PREFIX.with_name(PREFIX.name + "_min_degree_sensitivity.csv"),
        "figure_main": PREFIX.with_name(PREFIX.name + "_figure_main_recall10.csv"),
        "figure_niche": PREFIX.with_name(PREFIX.name + "_figure_niche_k10.csv"),
        "figure_runtime": PREFIX.with_name(PREFIX.name + "_figure_runtime.csv"),
    }
    dataset_stats.to_csv(outputs["dataset_stats"], index=False)
    main_results.to_csv(outputs["main_results"], index=False)
    ablations.to_csv(outputs["ablation_results"], index=False)
    stratified.to_csv(outputs["stratified_results"], index=False)
    runtime.to_csv(outputs["runtime_results"], index=False)
    min_degree.to_csv(outputs["min_degree_sensitivity"], index=False)
    main_fig.to_csv(outputs["figure_main"], index=False)
    niche_fig.to_csv(outputs["figure_niche"], index=False)
    runtime_fig.to_csv(outputs["figure_runtime"], index=False)

    md_path = write_markdown(dataset_stats, main_results, ablations, stratified, runtime, min_degree, figure_messages)
    print(f"Saved paper artifacts markdown to {md_path}")
    for name, path in outputs.items():
        print(f"Saved {name} to {path}")
    for message in figure_messages:
        print(f"Figure: {message}")


if __name__ == "__main__":
    main()
