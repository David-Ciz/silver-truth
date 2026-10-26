#!/usr/bin/env python3
"""Build compact MSC failure-diagnostic tables from completed artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MSC_DATASET = "Fluo-C2DL-MSC"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _phase2_rows(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    df = df[df["dataset"].eq(MSC_DATASET)].copy()
    df["source_table"] = "phase2_ablation_matrix"
    df["method_key"] = df["ablation_id"].astype(str)
    df["policy"] = ""
    df["fusion_backend"] = ""
    return df[
        [
            "method_key",
            "method_label",
            "row_family",
            "phase2_role",
            "coverage",
            "mean_iou",
            "mean_f1",
            "mean_validation_iou",
            "selected_methods",
            "policy",
            "fusion_backend",
            "source_table",
            "notes",
        ]
    ]


def _ranker_pruned_rows(path: Path, *, source_table: str) -> pd.DataFrame:
    df = _read_csv(path)
    df = df[df["dataset"].eq(MSC_DATASET)].copy()
    if df.empty:
        return pd.DataFrame()
    df["method_key"] = (
        "qa_pruned_"
        + df["fusion_backend"].astype(str).str.lower()
        + "__"
        + df["policy"].astype(str)
    )
    df["method_label"] = "QA-pruned " + df["fusion_backend"].astype(str)
    df["row_family"] = "full_pipeline"
    df["phase2_role"] = "qa_pruned_fusion"
    df["coverage"] = "all_available"
    df["mean_validation_iou"] = np.nan
    df["selected_methods"] = df["fusion_backend"].astype(str)
    df["source_table"] = source_table
    df["notes"] = (
        "Deployable QA/ranker-pruned fusion row; FUSION remains a baseline "
        "component, not an ENSEMBLE method."
    )
    return df[
        [
            "method_key",
            "method_label",
            "row_family",
            "phase2_role",
            "coverage",
            "mean_iou",
            "mean_f1",
            "mean_validation_iou",
            "selected_methods",
            "policy",
            "fusion_backend",
            "source_table",
            "notes",
        ]
    ]


def build_msc_method_comparison(
    *,
    phase2_dataset_summary: Path,
    simple_best_by_dataset: Path,
    non_simple_best_by_dataset: Path,
) -> pd.DataFrame:
    frames = [
        _phase2_rows(phase2_dataset_summary),
        _ranker_pruned_rows(
            simple_best_by_dataset,
            source_table="ranker_pruned_simple_best_policy",
        ),
        _ranker_pruned_rows(
            non_simple_best_by_dataset,
            source_table="ranker_pruned_non_simple_best_backend",
        ),
    ]
    table = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    table = table.drop_duplicates(
        subset=["method_key", "row_family", "source_table"], keep="first"
    )

    fusion_iou = _lookup_iou(table, "fusion_only_best")
    silver_iou = _lookup_iou(table, "silver_truth")
    best_competitor_iou = _lookup_iou(table, "best_competitor")
    table["delta_vs_fusion_only"] = table["mean_iou"] - fusion_iou
    table["delta_vs_silver_truth"] = table["mean_iou"] - silver_iou
    table["delta_vs_best_competitor"] = table["mean_iou"] - best_competitor_iou
    table["msc_interpretation"] = table.apply(_interpret_row, axis=1)
    return table.sort_values(
        ["mean_iou", "row_family", "method_label"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _lookup_iou(table: pd.DataFrame, method_key: str) -> float:
    match = table[table["method_key"].eq(method_key)]
    if match.empty:
        return float("nan")
    return float(pd.to_numeric(match.iloc[0]["mean_iou"], errors="coerce"))


def _interpret_row(row: pd.Series) -> str:
    key = str(row.get("method_key", ""))
    if key == "silver_truth":
        return "Official target context."
    if key == "best_competitor":
        return "External competitor context."
    if key == "fusion_only_best":
        return "External fusion-only baseline."
    if str(row.get("phase2_role", "")) == "qa_pruned_fusion":
        return "QA/ranker pruning plus fusion; strongest MSC deployable path in this table."
    if key == "no_source_prior":
        return "Removing source identity is a large MSC regression."
    if key == "no_qa":
        return "No-QA direct selection remains below fusion-only and QA-pruned fusion."
    if key == "full_image_vs_local_support_modes":
        return "Best direct soft-hybrid MSC selector, still below fusion-only."
    return "Direct candidate-selection or context row."


def write_markdown(table: pd.DataFrame, output_path: Path) -> None:
    display = table[
        [
            "method_label",
            "row_family",
            "mean_iou",
            "delta_vs_fusion_only",
            "delta_vs_silver_truth",
            "msc_interpretation",
        ]
    ].copy()
    lines = [
        "# MSC Failure Diagnostic Summary",
        "",
        "Dataset: `Fluo-C2DL-MSC`.",
        "",
        "FUSION rows are external/deployable pipeline context, not ENSEMBLE rows.",
        "",
        display.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2-dataset-summary",
        type=Path,
        default=Path(
            "hpc_results/soft_hybrid_probe/phase2_ablation_matrix/"
            "phase2_validation_selected_dataset_summary.csv"
        ),
    )
    parser.add_argument(
        "--simple-best-by-dataset",
        type=Path,
        default=Path(
            "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
            "all_available_simple_policy_best_by_dataset.csv"
        ),
    )
    parser.add_argument(
        "--non-simple-best-by-dataset",
        type=Path,
        default=Path(
            "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
            "non_simple_best_policy_backend_sweep/"
            "non_simple_best_policy_best_by_dataset.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpc_results/soft_hybrid_probe/msc_failure_diagnostics"),
    )
    args = parser.parse_args()

    table = build_msc_method_comparison(
        phase2_dataset_summary=args.phase2_dataset_summary,
        simple_best_by_dataset=args.simple_best_by_dataset,
        non_simple_best_by_dataset=args.non_simple_best_by_dataset,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "msc_method_comparison.csv"
    md_path = args.output_dir / "msc_failure_diagnostic_summary.md"
    table.to_csv(csv_path, index=False)
    write_markdown(table, md_path)
    print(f"Wrote MSC method comparison: {csv_path}")
    print(f"Wrote MSC diagnostic markdown: {md_path}")


if __name__ == "__main__":
    main()
