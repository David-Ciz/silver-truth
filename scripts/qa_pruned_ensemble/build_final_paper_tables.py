#!/usr/bin/env python3
"""Build final strict QA/ENSEMBLE paper tables with fold-level uncertainty."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from silver_truth.evaluation.reporting import (  # noqa: E402
    bootstrap_stratified_mean_ci,
    paired_wilcoxon_test,
)


DEFAULT_VALIDATION_DIR = (
    PROJECT_ROOT
    / "hpc_results"
    / "soft_hybrid_probe"
    / "qa_pruned_ensemble"
    / "validation_selection"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "hpc_results"
    / "soft_hybrid_probe"
    / "qa_pruned_ensemble"
    / "final_paper"
)
DEFAULT_SELECTION_RULES = (
    "global_validation_selected",
    "leave_one_dataset_mean_minus_1p000std_selected",
    "validation_mean_minus_2p000std_selected",
)
SIMPLE_CONTEXT = (
    {
        "label": "QA-pruned SIMPLE fixed source-balanced top3",
        "mean_iou": 0.8654,
        "scope": "external FUSION/deployable pipeline baseline",
    },
    {
        "label": "QA-pruned SIMPLE dataset-policy context",
        "mean_iou": 0.867534,
        "scope": "external FUSION/context row, not strict ENSEMBLE",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection-rule",
        action="append",
        default=None,
        help=(
            "Selection rule to include. Can be repeated. Defaults to the frozen "
            "strict QA/ENSEMBLE headline rules."
        ),
    )
    parser.add_argument("--reference-rule", default="global_validation_selected")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    selection_rules = tuple(args.selection_rule or DEFAULT_SELECTION_RULES)
    tables = build_final_tables(
        validation_dir=args.validation_dir,
        selection_rules=selection_rules,
        reference_rule=args.reference_rule,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_final_tables(args.output_dir, tables)
    print(f"Wrote final strict QA/ENSEMBLE paper tables to {args.output_dir}")
    print(tables["overall"].to_string(index=False))


def build_final_tables(
    *,
    validation_dir: Path,
    selection_rules: Sequence[str],
    reference_rule: str,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    fold_rows = pd.read_csv(
        validation_dir / "qa_pruned_ensemble_validation_fold_rows.csv"
    )
    selected_dataset_rows = load_selected_dataset_rows(
        validation_dir=validation_dir,
        selection_rules=selection_rules,
    )
    selected_fold_rows = collect_selected_fold_rows(
        fold_rows=fold_rows,
        selected_dataset_rows=selected_dataset_rows,
    )
    overall = summarize_selected_fold_rows(
        selected_fold_rows,
        selected_dataset_rows=selected_dataset_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    deltas = summarize_pairwise_deltas(
        selected_fold_rows,
        reference_rule=reference_rule,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    context = pd.DataFrame.from_records(SIMPLE_CONTEXT)
    return {
        "overall": overall,
        "dataset": selected_dataset_rows,
        "fold_rows": selected_fold_rows,
        "pairwise_deltas": deltas,
        "simple_context": context,
    }


def load_selected_dataset_rows(
    *,
    validation_dir: Path,
    selection_rules: Sequence[str],
) -> pd.DataFrame:
    source_paths = [
        validation_dir / "qa_pruned_ensemble_global_validation_selected_dataset_summary.csv",
        validation_dir / "qa_pruned_ensemble_stability_selected_dataset_summary.csv",
        validation_dir
        / "qa_pruned_ensemble_leave_one_dataset_validation_selected_dataset_summary.csv",
        validation_dir / "qa_pruned_ensemble_validation_selected_dataset_summary.csv",
        validation_dir / "qa_pruned_ensemble_guarded_validation_selected_dataset_summary.csv",
    ]
    frames = [pd.read_csv(path) for path in source_paths if path.exists()]
    if not frames:
        raise FileNotFoundError(f"No selected dataset summaries found in {validation_dir}")

    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["selection_rule"].isin(selection_rules)].copy()
    if rows.empty:
        raise ValueError(
            "No selected dataset rows matched requested rules: "
            + ", ".join(selection_rules)
        )
    rows = rows.drop_duplicates(["selection_rule", "dataset", "run_id"])
    return rows.sort_values(["selection_rule", "dataset"]).reset_index(drop=True)


def collect_selected_fold_rows(
    *,
    fold_rows: pd.DataFrame,
    selected_dataset_rows: pd.DataFrame,
) -> pd.DataFrame:
    required = ["selection_rule", "dataset", "run_id"]
    missing = [column for column in required if column not in selected_dataset_rows.columns]
    if missing:
        raise ValueError(f"Selected dataset rows missing columns: {missing}")

    test_folds = fold_rows[fold_rows["split"].eq("test")].copy()
    selectors = selected_dataset_rows[required].drop_duplicates()
    selected = selectors.merge(test_folds, on=["dataset", "run_id"], how="left")
    missing_rows = selected[selected["fold"].isna()][required].drop_duplicates()
    if not missing_rows.empty:
        raise ValueError(
            "Selected runs were not found in test fold rows:\n"
            + missing_rows.to_string(index=False)
        )
    return selected.sort_values(["selection_rule", "dataset", "fold"]).reset_index(
        drop=True
    )


def summarize_selected_fold_rows(
    selected_fold_rows: pd.DataFrame,
    *,
    selected_dataset_rows: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for rule, group in selected_fold_rows.groupby("selection_rule", sort=False):
        iou_ci = _stratified_ci(
            group,
            metric="mean_iou",
            n_resamples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        f1_ci = _stratified_ci(
            group,
            metric="mean_f1",
            n_resamples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        )
        dataset_selection = selected_dataset_rows[
            selected_dataset_rows["selection_rule"].eq(rule)
        ]
        records.append(
            {
                "selection_rule": rule,
                "mean_iou": iou_ci["mean"],
                "iou_ci_lower": iou_ci["ci_lower"],
                "iou_ci_upper": iou_ci["ci_upper"],
                "mean_f1": f1_ci["mean"],
                "f1_ci_lower": f1_ci["ci_lower"],
                "f1_ci_upper": f1_ci["ci_upper"],
                "n_datasets": int(group["dataset"].nunique()),
                "n_folds": int(len(group)),
                "n_cells": int(group["n_cells"].sum()),
                "selected_run_ids": ";".join(sorted(group["run_id"].astype(str).unique())),
                "method_label": _compact_label(dataset_selection, "method_label"),
                "ensemble_variant": _compact_label(dataset_selection, "ensemble_variant"),
                "soft_gate": _compact_label(dataset_selection, "soft_gate"),
                "threshold": _compact_label(dataset_selection, "threshold"),
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values("mean_iou", ascending=False)
        .reset_index(drop=True)
    )


def summarize_pairwise_deltas(
    selected_fold_rows: pd.DataFrame,
    *,
    reference_rule: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    reference = selected_fold_rows[
        selected_fold_rows["selection_rule"].eq(reference_rule)
    ][["dataset", "fold", "mean_iou", "mean_f1"]].rename(
        columns={"mean_iou": "reference_iou", "mean_f1": "reference_f1"}
    )
    if reference.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for rule, group in selected_fold_rows.groupby("selection_rule", sort=False):
        if rule == reference_rule:
            continue
        candidate = group[["dataset", "fold", "mean_iou", "mean_f1"]].rename(
            columns={"mean_iou": "candidate_iou", "mean_f1": "candidate_f1"}
        )
        paired = reference.merge(candidate, on=["dataset", "fold"], how="inner")
        if paired.empty:
            continue
        paired["delta_iou"] = paired["candidate_iou"] - paired["reference_iou"]
        paired["delta_f1"] = paired["candidate_f1"] - paired["reference_f1"]
        delta_iou_ci = _stratified_ci(
            paired,
            metric="delta_iou",
            n_resamples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        delta_f1_ci = _stratified_ci(
            paired,
            metric="delta_f1",
            n_resamples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        )
        test = paired_wilcoxon_test(
            paired["reference_iou"],
            paired["candidate_iou"],
        )
        records.append(
            {
                "candidate_rule": rule,
                "reference_rule": reference_rule,
                "delta_iou": delta_iou_ci["mean"],
                "delta_iou_ci_lower": delta_iou_ci["ci_lower"],
                "delta_iou_ci_upper": delta_iou_ci["ci_upper"],
                "delta_f1": delta_f1_ci["mean"],
                "delta_f1_ci_lower": delta_f1_ci["ci_lower"],
                "delta_f1_ci_upper": delta_f1_ci["ci_upper"],
                "n_pairs": int(test["n_pairs"]),
                "wilcoxon_p_value_iou": float(test["p_value"]),
                "wilcoxon_note": str(test["note"]),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        "delta_iou", ascending=False
    )


def write_final_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["overall"].to_csv(
        output_dir / "strict_qa_ensemble_final_overall.csv",
        index=False,
    )
    tables["dataset"].to_csv(
        output_dir / "strict_qa_ensemble_final_dataset.csv",
        index=False,
    )
    tables["fold_rows"].to_csv(
        output_dir / "strict_qa_ensemble_final_fold_rows.csv",
        index=False,
    )
    tables["pairwise_deltas"].to_csv(
        output_dir / "strict_qa_ensemble_pairwise_deltas.csv",
        index=False,
    )
    tables["simple_context"].to_csv(
        output_dir / "external_simple_context.csv",
        index=False,
    )
    write_markdown_summary(output_dir / "strict_qa_ensemble_final_summary.md", tables)


def write_markdown_summary(
    output_path: Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# Strict QA/ENSEMBLE Final Paper Tables",
        "",
        "Scope: strict QA and ENSEMBLE rows only. FUSION/SIMPLE is external context, not our ENSEMBLE method.",
        "",
        "## Final Strict Rows",
        "",
    ]
    lines.extend(_markdown_table(tables["overall"]))
    lines.extend(["", "## Pairwise Deltas Vs Global Validation Selected", ""])
    lines.extend(_markdown_table(tables["pairwise_deltas"]))
    lines.extend(["", "## External SIMPLE Context", ""])
    lines.extend(_markdown_table(tables["simple_context"]))
    lines.extend(
        [
            "",
            "## Claim Guidance",
            "",
            "Use the global validation-selected row as the clean conservative strict ENSEMBLE result.",
            "Use the leave-one-dataset stability row as the conservative stability-selected follow-up.",
            "Use the full-panel mean-minus-2std row as the strongest strict diagnostic, with the caveat that the margin over QA-pruned SIMPLE context is tiny.",
            "Do not write that ENSEMBLE definitively beats SIMPLE unless the paper table explicitly limits the claim and discusses uncertainty.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stratified_ci(
    frame: pd.DataFrame,
    *,
    metric: str,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    values_by_dataset = {
        str(dataset): group[metric]
        for dataset, group in frame.groupby("dataset", sort=True)
    }
    return bootstrap_stratified_mean_ci(
        values_by_dataset,
        n_resamples=n_resamples,
        seed=seed,
    )


def _compact_label(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = sorted({str(value) for value in frame[column].dropna().unique()})
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "mixed: " + "; ".join(values)


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    shown = frame.copy().fillna("")
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    return shown.to_markdown(index=False).splitlines()


if __name__ == "__main__":
    main()
