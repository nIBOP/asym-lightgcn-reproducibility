from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATE_TAG = "2026-05-14"
THRESHOLDS = [4, 5, 6, 8, 10]
DATASET_NAMES = {threshold: f"clean_amazon_reduced_min{threshold}" for threshold in THRESHOLDS}
DATASET_DIRS = {
    threshold: Path("reduced_datasets/cold_preserving") / DATASET_NAMES[threshold]
    for threshold in THRESHOLDS
}

THRESHOLD_SUMMARY = Path("train_logs/amazon_threshold_sweep_2026-05-14_10-06-17_summary.csv")
MIN5_SUMMARY = Path("train_logs/amazon_min5_six_seed_summary_2026-05-08_summary.csv")
MIN5_STRATA = Path("train_logs/amazon_min5_item_strata_six_seed_summary_2026-05-08.csv")
MOSTPOP_THRESHOLD = Path("train_logs/mostpop_amazon_threshold_sweep_2026-05-14.csv")
MOSTPOP_MIN5 = Path("train_logs/mostpop_amazon_min_degree_6seed_2026-05-08.csv")

OUT_DIR = Path("train_logs")
FIG_DIR = Path("paper/figures")


def read_inter(dataset_dir, dataset_name):
    path = dataset_dir / f"{dataset_name}.inter"
    return pd.read_csv(path, sep="\t")


def raw_dataset_stats():
    rows = []
    for threshold in THRESHOLDS:
        dataset_name = DATASET_NAMES[threshold]
        frame = read_inter(DATASET_DIRS[threshold], dataset_name)
        users = frame["user_id:token"].nunique()
        items = frame["item_id:token"].nunique()
        interactions = len(frame)
        user_degree = frame.groupby("user_id:token").size()
        item_degree = frame.groupby("item_id:token").size()
        rows.append(
            {
                "Dataset": f"Amazon min{threshold}",
                "Min user degree": threshold,
                "Users": int(users),
                "Items": int(items),
                "Interactions": int(interactions),
                "Avg interactions/user": interactions / users,
                "Avg interactions/item": interactions / items,
                "Density": interactions / (users * items),
                "Min user interactions": int(user_degree.min()),
                "Median user interactions": float(user_degree.median()),
                "Max user interactions": int(user_degree.max()),
                "Median item interactions": float(item_degree.median()),
            }
        )
    return pd.DataFrame(rows)


def split_item_stats():
    frames = []
    for path in [MOSTPOP_THRESHOLD, MOSTPOP_MIN5]:
        if path.exists():
            frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = frame[
        frame["Dataset"].isin(DATASET_NAMES.values())
        & frame["Model"].eq("MostPopIndependent")
        & frame["Split"].eq("test")
        & frame["TopK"].astype(int).eq(10)
    ].copy()
    records = []
    for dataset, group in frame.groupby("Dataset"):
        threshold = int(dataset.rsplit("min", 1)[1])
        record = {"Dataset": f"Amazon min{threshold}"}
        for column in [
            "Users",
            "Train Interactions",
            "Train Item Count",
            "Head Item Count",
            "Torso Item Count",
            "Tail Item Count",
            "Cold Item Count",
        ]:
            values = pd.to_numeric(group[column], errors="coerce")
            record[f"{column} mean"] = values.mean()
            record[f"{column} std"] = values.std(ddof=1)
        records.append(record)
    return pd.DataFrame(records)


def fmt_mean_std(row, column):
    mean = row.get(f"{column} mean")
    std = row.get(f"{column} std")
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.0f}"
    return f"{mean:.0f} +/- {std:.0f}"


def write_dataset_stats():
    raw = raw_dataset_stats()
    split = split_item_stats()
    stats = raw.merge(split, on="Dataset", how="left")
    csv_path = OUT_DIR / f"amazon_threshold_dataset_stats_{DATE_TAG}.csv"
    md_path = OUT_DIR / f"amazon_threshold_dataset_stats_{DATE_TAG}.md"
    stats.to_csv(csv_path, index=False)

    printable = stats.copy()
    for column in ["Avg interactions/user", "Avg interactions/item"]:
        printable[column] = printable[column].map(lambda value: f"{value:.3f}")
    printable["Density"] = printable["Density"].map(lambda value: f"{value:.6f}")
    for column in [
        "Train Interactions",
        "Train Item Count",
        "Head Item Count",
        "Torso Item Count",
        "Tail Item Count",
        "Cold Item Count",
    ]:
        printable[column] = printable.apply(lambda row, col=column: fmt_mean_std(row, col), axis=1)
        printable = printable.drop(columns=[f"{column} mean", f"{column} std"])
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Amazon Threshold Dataset Statistics\n\n")
        handle.write(
            "Raw counts are computed from the filtered interaction files. "
            "Train/head/torso/tail/cold counts are averaged over available random split seeds.\n\n"
        )
        handle.write(printable.to_markdown(index=False))
        handle.write("\n")
    return stats, csv_path, md_path


def metric_row(dataset, model, metric):
    row = dataset[dataset["Model"].eq(model)]
    if row.empty:
        return None
    return float(row.iloc[0][f"{metric} mean"])


def load_threshold_quality():
    threshold = pd.read_csv(THRESHOLD_SUMMARY)
    min5 = pd.read_csv(MIN5_SUMMARY)
    rows = []
    for threshold_value in THRESHOLDS:
        dataset_name = DATASET_NAMES[threshold_value]
        source = min5 if threshold_value == 5 else threshold
        dataset_rows = source[source["Dataset"].eq(dataset_name)]
        for model in ["Semantic-only", "SemanticGatedBPR", "LightGCN", "NCL", "AsymLightGCN"]:
            value = metric_row(dataset_rows, model, "recall@50")
            if value is not None:
                rows.append(
                    {
                        "Threshold": threshold_value,
                        "Dataset": f"Amazon min{threshold_value}",
                        "Model": model,
                        "Metric": "Recall@50",
                        "Value": value,
                    }
                )
    return pd.DataFrame(rows)


def load_threshold_tail():
    threshold = pd.read_csv(THRESHOLD_SUMMARY)
    min5 = pd.read_csv(MIN5_STRATA)
    rows = []
    for threshold_value in THRESHOLDS:
        dataset_name = DATASET_NAMES[threshold_value]
        if threshold_value == 5:
            dataset_rows = min5[(min5["Dataset"].eq(dataset_name)) & (min5["TopK"].astype(int).eq(10))]
            for model in ["LightGCN", "NCL", "AsymLightGCN"]:
                row = dataset_rows[dataset_rows["Model"].eq(model)]
                if row.empty:
                    continue
                value = float(row.iloc[0]["recall_tail_mean"])
                rows.append(
                    {
                        "Threshold": threshold_value,
                        "Dataset": f"Amazon min{threshold_value}",
                        "Model": model,
                        "Metric": "Tail Recall@10",
                        "Value": value,
                    }
                )
        else:
            dataset_rows = threshold[threshold["Dataset"].eq(dataset_name)]
            for model in ["SemanticGatedBPR", "LightGCN", "NCL", "AsymLightGCN"]:
                row = dataset_rows[dataset_rows["Model"].eq(model)]
                if row.empty:
                    continue
                value = float(row.iloc[0]["recall_tail mean"])
                rows.append(
                    {
                        "Threshold": threshold_value,
                        "Dataset": f"Amazon min{threshold_value}",
                        "Model": model,
                        "Metric": "Tail Recall@10",
                        "Value": value,
                    }
                )
    return pd.DataFrame(rows)


def plot_threshold_sweep():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    quality = load_threshold_quality()
    tail = load_threshold_tail()
    plot_data = pd.concat([quality, tail], ignore_index=True, sort=False)
    data_path = OUT_DIR / f"amazon_threshold_figure_data_{DATE_TAG}.csv"
    plot_data.to_csv(data_path, index=False)

    colors = {
        "AsymLightGCN": "#1f77b4",
        "LightGCN": "#d62728",
        "NCL": "#2ca02c",
        "Semantic-only": "#9467bd",
        "SemanticGatedBPR": "#ff7f0e",
    }
    markers = {
        "AsymLightGCN": "o",
        "LightGCN": "s",
        "NCL": "^",
        "Semantic-only": "D",
        "SemanticGatedBPR": "P",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    for axis, metric in zip(axes, ["Recall@50", "Tail Recall@10"]):
        subset = plot_data[plot_data["Metric"].eq(metric)]
        for model in ["AsymLightGCN", "LightGCN", "NCL", "SemanticGatedBPR", "Semantic-only"]:
            group = subset[subset["Model"].eq(model)].sort_values("Threshold")
            if group.empty:
                continue
            axis.plot(
                group["Threshold"],
                group["Value"],
                marker=markers.get(model, "o"),
                linewidth=2,
                markersize=5,
                label=model,
                color=colors.get(model),
            )
        axis.set_title(metric)
        axis.set_xlabel("Amazon min-degree threshold")
        axis.set_ylabel(metric)
        axis.set_xticks(THRESHOLDS)
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()

    png_path = FIG_DIR / f"amazon_threshold_sweep_{DATE_TAG}.png"
    pdf_path = FIG_DIR / f"amazon_threshold_sweep_{DATE_TAG}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return data_path, png_path, pdf_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats, stats_csv, stats_md = write_dataset_stats()
    figure_data, png_path, pdf_path = plot_threshold_sweep()
    print(f"Saved dataset stats to {stats_csv}")
    print(f"Saved dataset stats report to {stats_md}")
    print(f"Saved figure data to {figure_data}")
    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {pdf_path}")
    print(stats[["Dataset", "Users", "Items", "Interactions", "Avg interactions/user"]].to_string(index=False))


if __name__ == "__main__":
    main()
