"""Small paired-seed significance helper for paper-facing result tables.

Most current rows have three paired seeds, while the Amazon min5 core block
has been extended to six seeds. Therefore the output should be treated as a
directional robustness check, not as a decisive hypothesis test. We report
exact paired sign-flip p-values and t-based confidence intervals for the
paired mean delta.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("recall@10", "ndcg@10", "recall@50", "ndcg@50")
T_CRIT_95 = {
    2: 12.7062047364,
    3: 4.3026527299,
    4: 3.1824463053,
    5: 2.7764451052,
    6: 2.5705818366,
    7: 2.4469118511,
    8: 2.3646242510,
    9: 2.3060041350,
    10: 2.2621571628,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="train_logs/paper_facing_reduced_results_corrected_2026-05-07_selected_rows.csv",
        help="Base per-seed paper-facing result CSV.",
    )
    parser.add_argument(
        "--amazon-min5-input",
        default="train_logs/amazon_min5_six_seed_summary_2026-05-08_selected_rows.csv",
        help="Optional Amazon min5 six-seed selected rows that supersede the base Amazon min5 rows.",
    )
    parser.add_argument(
        "--output-prefix",
        default="train_logs/paper_significance_2026-05-08",
        help="Output prefix for .csv and .md files.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_combined_rows(base_path: Path, amazon_min5_path: Path | None) -> list[dict[str, str]]:
    rows = load_rows(base_path)
    if amazon_min5_path is None or not amazon_min5_path.exists():
        return rows
    rows = [row for row in rows if row.get("Dataset") != "clean_amazon_reduced_min5"]
    rows.extend(load_rows(amazon_min5_path))
    return rows


def paired_signflip_p(deltas: list[float]) -> float | None:
    if not deltas:
        return None
    observed = abs(sum(deltas) / len(deltas))
    total = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        stat = abs(sum(sign * delta for sign, delta in zip(signs, deltas)) / len(deltas))
        total += 1
        if stat >= observed - 1e-15:
            extreme += 1
    return extreme / total


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def ci95(values: list[float]) -> tuple[float | None, float | None]:
    n = len(values)
    if n < 2:
        return None, None
    tcrit = T_CRIT_95.get(n)
    if tcrit is None:
        # Normal approximation fallback keeps the helper useful for future
        # larger seed sets outside the current n=3/n=6 package.
        tcrit = 1.96
    half_width = tcrit * sample_std(values) / math.sqrt(n)
    m = mean(values)
    return m - half_width, m + half_width


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def fmt_p(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def build_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[int, dict[str, str]]]:
    index: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        dataset = row["Dataset"]
        model = row["Model"]
        seed = int(row["Seed"])
        index[(dataset, model)][seed] = row
    return index


def compare(
    index: dict[tuple[str, str], dict[int, dict[str, str]]],
    dataset: str,
    anchor: str,
    baseline: str,
) -> list[dict[str, str]]:
    anchor_rows = index.get((dataset, anchor), {})
    baseline_rows = index.get((dataset, baseline), {})
    seeds = sorted(set(anchor_rows) & set(baseline_rows))
    output = []
    for metric in METRICS:
        anchor_values = [float(anchor_rows[seed][metric]) for seed in seeds]
        baseline_values = [float(baseline_rows[seed][metric]) for seed in seeds]
        deltas = [a - b for a, b in zip(anchor_values, baseline_values)]
        baseline_mean = mean(baseline_values) if baseline_values else float("nan")
        delta_mean = mean(deltas) if deltas else float("nan")
        rel_delta = delta_mean / baseline_mean * 100 if baseline_mean else float("nan")
        ci_low, ci_high = ci95(deltas)
        output.append(
            {
                "Dataset": dataset,
                "Anchor": anchor,
                "Baseline": baseline,
                "Metric": metric,
                "Seeds": "/".join(str(seed) for seed in seeds),
                "N": str(len(seeds)),
                "Anchor Mean": fmt_float(mean(anchor_values) if anchor_values else None, 6),
                "Baseline Mean": fmt_float(baseline_mean if baseline_values else None, 6),
                "Mean Delta": fmt_float(delta_mean if deltas else None, 6),
                "Relative Delta %": fmt_float(rel_delta if deltas else None, 2),
                "Delta CI95 Low": fmt_float(ci_low, 6),
                "Delta CI95 High": fmt_float(ci_high, 6),
                "Signflip p": fmt_p(paired_signflip_p(deltas)),
            }
        )
    return output


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    columns = [
        "Dataset",
        "Anchor",
        "Baseline",
        "Metric",
        "N",
        "Mean Delta",
        "Relative Delta %",
        "Delta CI95 Low",
        "Delta CI95 High",
        "Signflip p",
    ]
    lines = [
        "# Paper Significance Sanity Check",
        "",
        "This table uses paired seeds and exact sign-flip p-values. Amazon min5 uses",
        "six paired seeds where available, so an all-same-direction comparison can",
        "reach a two-sided p-value of 0.031. Movies/Yelp comparisons still use three",
        "paired seeds, where the minimum possible two-sided p-value is 0.250. These",
        "rows should therefore be read as directional robustness checks rather than",
        "formal proof of significance.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_combined_rows(Path(args.input), Path(args.amazon_min5_input))
    index = build_index(rows)

    comparisons = [
        ("clean_amazon_reduced_min5", "AsymLightGCN", "LightGCN"),
        ("clean_amazon_reduced_min5", "AsymLightGCN", "NCL"),
        ("clean_amazon_reduced_min5", "AsymLightGCN", "BPR"),
        ("clean_amazon_reduced_min5", "AsymLightGCN", "MostPopIndependent"),
        ("clean_amazon_reduced_min5", "AsymLightGCN", "Semantic-only"),
        ("clean_movies_reduced", "AsymLightGCN", "LightGCN"),
        ("clean_movies_reduced", "AsymLightGCN", "BPR"),
        ("clean_yelp_reduced", "AsymLightGCN", "LightGCN"),
        ("clean_yelp_reduced", "AsymLightGCN", "MostPopIndependent"),
    ]
    out_rows: list[dict[str, str]] = []
    for dataset, anchor, baseline in comparisons:
        out_rows.extend(compare(index, dataset, anchor, baseline))

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_rows, output_prefix.with_suffix(".csv"))
    write_markdown(out_rows, output_prefix.with_suffix(".md"))
    print(f"Wrote {len(out_rows)} rows to {output_prefix.with_suffix('.csv')}")
    print(f"Wrote markdown summary to {output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
