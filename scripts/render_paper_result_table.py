#!/usr/bin/env python3
"""Render canonical paper result summaries as a Markdown table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DISPLAY_COLUMNS = [
    ("dataset_label", "Dataset"),
    ("fold_label", "Fold"),
    ("best_fusion_iou", "Best fusion-only"),
    ("best_competitor_iou", "Best competitor"),
    ("median_competitor_iou", "Median competitor"),
    ("worst_competitor_iou", "Worst competitor"),
    ("silver_truth_iou", "Silver truth"),
    ("best_ours_iou", "Best ours"),
    ("best_ours_method", "Ours method"),
]


def _format_score(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    return f"{float(number):.4f}"


def render_table(summary_csv: Path) -> str:
    summary = pd.read_csv(summary_csv)
    required = [
        "dataset",
        "crop_tag",
        "fold",
        *[source for source, _ in DISPLAY_COLUMNS if source.endswith("_iou")],
        "best_ours_method",
    ]
    missing = [source for source in required if source not in summary.columns]
    if missing:
        raise ValueError(f"{summary_csv} is missing columns: {', '.join(missing)}")

    summary["dataset_label"] = summary.apply(
        lambda row: f"{row['dataset']} ({row['crop_tag']})", axis=1
    )
    summary["fold_label"] = summary["fold"].map(
        lambda value: str(value)
        .removeprefix("fold-")
        .replace("mean", "Mean")
    )
    display = summary[[source for source, _ in DISPLAY_COLUMNS]].copy()
    score_columns = [source for source, _ in DISPLAY_COLUMNS if source.endswith("_iou")]
    for column in score_columns:
        display[column] = display[column].map(_format_score)
    display = display.rename(columns=dict(DISPLAY_COLUMNS))
    return display.to_markdown(index=False, disable_numparse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a paper_result_summary.csv file as Markdown."
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Markdown output path. Defaults to <summary_csv>.md.",
    )
    args = parser.parse_args()

    table = render_table(args.summary_csv)
    output = args.output or args.summary_csv.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table + "\n")
    print(table)
    print(f"\nWrote Markdown table: {output}")


if __name__ == "__main__":
    main()
