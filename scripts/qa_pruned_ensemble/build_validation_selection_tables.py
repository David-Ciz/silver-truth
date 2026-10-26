#!/usr/bin/env python3
"""Build validation-selection tables for strict QA-pruned ENSEMBLE rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_OUTPUT_ROOT, filter_strict_ensemble_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-glob",
        action="append",
        default=None,
        help=(
            "Glob for weighted_object_vote_summary.csv files. Defaults to all "
            "QA-pruned ENSEMBLE hard/soft/sweep summaries."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "validation_selection",
    )
    parser.add_argument(
        "--guard-margin",
        action="append",
        type=float,
        default=None,
        help=(
            "Validation IoU margin for guarded per-dataset switching away from "
            "the global validation-selected run. Can be repeated."
        ),
    )
    parser.add_argument(
        "--stability-std-penalty",
        action="append",
        type=float,
        default=None,
        help=(
            "Penalty multiplier for full-coverage validation-stability selection: "
            "score = validation_mean_iou - penalty * validation_std_iou."
        ),
    )
    args = parser.parse_args()

    patterns = args.summary_glob or [
        str(DEFAULT_OUTPUT_ROOT / "hard_gated_object_vote/**/weighted_object_vote_summary.csv"),
        str(DEFAULT_OUTPUT_ROOT / "soft_gated_object_vote/**/weighted_object_vote_summary.csv"),
        str(DEFAULT_OUTPUT_ROOT / "soft_gate_sweep/**/weighted_object_vote_summary.csv"),
        str(DEFAULT_OUTPUT_ROOT / "learned_candidate_set_weight/**/weighted_object_vote_summary.csv"),
    ]
    rows = collect_summary_rows(patterns)
    if rows.empty:
        raise SystemExit("No strict QA-pruned ENSEMBLE summary rows matched.")

    dataset_summary = summarize_dataset(rows)
    fixed = fixed_test_overall(dataset_summary)
    validation_selected = select_by_validation(dataset_summary)
    global_selected = select_by_global_validation(dataset_summary)
    stability_mean_minus_std_selected = stability_penalty_outputs(
        dataset_summary,
        penalties=tuple(args.stability_std_penalty or (1.0,)),
    )
    leave_one_dataset_stability_selected = leave_one_dataset_stability_penalty_outputs(
        dataset_summary,
        penalties=tuple(args.stability_std_penalty or (1.0,)),
    )
    stability_maximin_selected = select_by_validation_maximin(dataset_summary)
    leave_one_dataset_selected = select_by_leave_one_dataset_validation(dataset_summary)
    guarded_selected = guarded_margin_outputs(
        dataset_summary,
        global_selected=global_selected,
        margins=tuple(args.guard_margin or (0.0, 0.005, 0.01, 0.02)),
    )
    selected_overall = summarize_selected_outputs(
        {
            "validation_selected_by_dataset": validation_selected,
            "global_validation_selected": global_selected,
            "validation_maximin_selected": stability_maximin_selected,
            **stability_mean_minus_std_selected,
            **leave_one_dataset_stability_selected,
            "leave_one_dataset_validation_selected": leave_one_dataset_selected,
            **guarded_selected,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "qa_pruned_ensemble_validation_fold_rows.csv", index=False)
    dataset_summary.to_csv(
        args.output_dir / "qa_pruned_ensemble_validation_dataset_summary.csv",
        index=False,
    )
    fixed.to_csv(
        args.output_dir / "qa_pruned_ensemble_fixed_test_overall_summary.csv",
        index=False,
    )
    validation_selected.to_csv(
        args.output_dir / "qa_pruned_ensemble_validation_selected_dataset_summary.csv",
        index=False,
    )
    global_selected.to_csv(
        args.output_dir / "qa_pruned_ensemble_global_validation_selected_dataset_summary.csv",
        index=False,
    )
    stability_frames = [
        frame
        for frame in [
            stability_maximin_selected,
            *stability_mean_minus_std_selected.values(),
            *leave_one_dataset_stability_selected.values(),
        ]
        if not frame.empty
    ]
    if stability_frames:
        pd.concat(stability_frames, ignore_index=True).to_csv(
            args.output_dir
            / "qa_pruned_ensemble_stability_selected_dataset_summary.csv",
            index=False,
        )
    leave_one_dataset_selected.to_csv(
        args.output_dir
        / "qa_pruned_ensemble_leave_one_dataset_validation_selected_dataset_summary.csv",
        index=False,
    )
    if guarded_selected:
        pd.concat(guarded_selected.values(), ignore_index=True).to_csv(
            args.output_dir
            / "qa_pruned_ensemble_guarded_validation_selected_dataset_summary.csv",
            index=False,
        )
    selected_overall.to_csv(
        args.output_dir / "qa_pruned_ensemble_validation_selection_overall_summary.csv",
        index=False,
    )
    write_markdown_report(
        output_path=args.output_dir / "qa_pruned_ensemble_validation_selection_summary.md",
        fixed=fixed,
        selected_overall=selected_overall,
        validation_selected=validation_selected,
        global_selected=global_selected,
        stability_selected={
            "validation_maximin_selected": stability_maximin_selected,
            **stability_mean_minus_std_selected,
            **leave_one_dataset_stability_selected,
        },
        leave_one_dataset_selected=leave_one_dataset_selected,
        guarded_selected=guarded_selected,
    )
    print(f"Wrote QA-pruned ENSEMBLE validation-selection tables to {args.output_dir}")
    print(selected_overall.to_string(index=False))


def collect_summary_rows(patterns: Sequence[str]) -> pd.DataFrame:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path().glob(pattern))
    frames: list[pd.DataFrame] = []
    for path in sorted(set(paths)):
        frame = pd.read_csv(path)
        frame["summary_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    rows = filter_strict_ensemble_rows(pd.concat(frames, ignore_index=True))
    if rows.empty:
        return rows
    rows["mean_iou"] = pd.to_numeric(rows["mean_iou"], errors="coerce")
    rows["mean_f1"] = pd.to_numeric(rows["mean_f1"], errors="coerce")
    rows["run_id"] = rows.apply(_run_id, axis=1)
    return rows


def summarize_dataset(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "run_id",
        "result_family",
        "row_family",
        "method_key",
        "method_label",
        "ensemble_variant",
        "qa_pruning_policy",
        "soft_gate",
        "dataset",
        "crop_tag",
        "variant",
        "split",
        "weights_column",
        "boundary_support_column",
        "boundary_floor",
        "boundary_floor_column",
        "boundary_iterations",
        "top_k_per_group",
        "threshold",
    ]
    present = [column for column in group_cols if column in rows.columns]
    return (
        rows.groupby(present, dropna=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            n_folds=("fold", "nunique"),
            n_cells=("n_cells", "sum"),
            mean_candidates_used=("mean_candidates_used", "mean"),
        )
        .reset_index()
        .sort_values(["split", "dataset", "mean_iou"], ascending=[True, True, False])
    )


def fixed_test_overall(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    test = dataset_summary[dataset_summary["split"].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    group_cols = [
        "run_id",
        "result_family",
        "row_family",
        "method_key",
        "method_label",
        "ensemble_variant",
        "qa_pruning_policy",
        "soft_gate",
        "weights_column",
        "boundary_support_column",
        "boundary_floor",
        "boundary_floor_column",
        "boundary_iterations",
        "threshold",
    ]
    present = [column for column in group_cols if column in test.columns]
    overall = (
        test.groupby(present, dropna=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            n_datasets=("dataset", "nunique"),
            n_folds=("n_folds", "sum"),
            n_cells=("n_cells", "sum"),
            mean_candidates_used=("mean_candidates_used", "mean"),
        )
        .reset_index()
        .sort_values("mean_iou", ascending=False)
    )
    overall["selection_rule"] = "fixed_run"
    return overall.reset_index(drop=True)


def select_by_validation(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty:
        return pd.DataFrame()
    selected_runs = (
        validation.sort_values(
            ["dataset", "mean_iou", "mean_f1", "run_id"],
            ascending=[True, False, False, True],
        )
        .groupby("dataset", dropna=False)
        .head(1)[["dataset", "run_id", "mean_iou"]]
        .rename(columns={"mean_iou": "validation_mean_iou"})
    )
    selected = test.merge(selected_runs, on=["dataset", "run_id"], how="inner")
    selected["selection_rule"] = "validation_selected_by_dataset"
    return selected.sort_values("dataset").reset_index(drop=True)


def select_by_global_validation(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty:
        return pd.DataFrame()
    eligible = _runs_with_dataset_coverage(
        validation,
        required_datasets=validation["dataset"].dropna().astype(str).unique(),
    )
    eligible &= _runs_with_dataset_coverage(
        test,
        required_datasets=test["dataset"].dropna().astype(str).unique(),
    )
    validation = validation[validation["run_id"].isin(eligible)]
    test = test[test["run_id"].isin(eligible)]
    if validation.empty or test.empty:
        return pd.DataFrame()
    selected_run = (
        validation.groupby("run_id", dropna=False)
        .agg(validation_mean_iou=("mean_iou", "mean"), validation_mean_f1=("mean_f1", "mean"))
        .reset_index()
        .sort_values(["validation_mean_iou", "validation_mean_f1", "run_id"], ascending=[False, False, True])
        .head(1)[["run_id", "validation_mean_iou"]]
    )
    selected = test.merge(selected_run, on="run_id", how="inner")
    selected["selection_rule"] = "global_validation_selected"
    return selected.sort_values("dataset").reset_index(drop=True)


def stability_penalty_outputs(
    dataset_summary: pd.DataFrame,
    *,
    penalties: Sequence[float],
) -> dict[str, pd.DataFrame]:
    return {
        f"validation_mean_minus_{_margin_token(penalty)}std_selected": (
            select_by_validation_mean_minus_std(
                dataset_summary,
                std_penalty=penalty,
            )
        )
        for penalty in penalties
    }


def leave_one_dataset_stability_penalty_outputs(
    dataset_summary: pd.DataFrame,
    *,
    penalties: Sequence[float],
) -> dict[str, pd.DataFrame]:
    return {
        f"leave_one_dataset_mean_minus_{_margin_token(penalty)}std_selected": (
            select_by_leave_one_dataset_validation_mean_minus_std(
                dataset_summary,
                std_penalty=penalty,
            )
        )
        for penalty in penalties
    }


def select_by_validation_mean_minus_std(
    dataset_summary: pd.DataFrame,
    *,
    std_penalty: float,
) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty:
        return pd.DataFrame()
    validation, test = _full_coverage_validation_test(validation, test)
    if validation.empty or test.empty:
        return pd.DataFrame()

    scores = (
        validation.groupby("run_id", dropna=False)
        .agg(
            validation_mean_iou=("mean_iou", "mean"),
            validation_std_iou=("mean_iou", lambda values: values.std(ddof=0)),
            validation_min_iou=("mean_iou", "min"),
            validation_mean_f1=("mean_f1", "mean"),
        )
        .reset_index()
    )
    scores["validation_stability_score"] = (
        scores["validation_mean_iou"] - std_penalty * scores["validation_std_iou"]
    )
    selected_run = (
        scores.sort_values(
            [
                "validation_stability_score",
                "validation_mean_iou",
                "validation_mean_f1",
                "run_id",
            ],
            ascending=[False, False, False, True],
        )
        .head(1)
    )
    selected = test.merge(selected_run, on="run_id", how="inner")
    selected["validation_std_penalty"] = std_penalty
    selected["selection_rule"] = (
        f"validation_mean_minus_{_margin_token(std_penalty)}std_selected"
    )
    return selected.sort_values("dataset").reset_index(drop=True)


def select_by_validation_maximin(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty:
        return pd.DataFrame()
    validation, test = _full_coverage_validation_test(validation, test)
    if validation.empty or test.empty:
        return pd.DataFrame()

    selected_run = (
        validation.groupby("run_id", dropna=False)
        .agg(
            validation_mean_iou=("mean_iou", "mean"),
            validation_min_iou=("mean_iou", "min"),
            validation_mean_f1=("mean_f1", "mean"),
        )
        .reset_index()
        .sort_values(
            ["validation_min_iou", "validation_mean_iou", "validation_mean_f1", "run_id"],
            ascending=[False, False, False, True],
        )
        .head(1)
    )
    selected = test.merge(selected_run, on="run_id", how="inner")
    selected["selection_rule"] = "validation_maximin_selected"
    return selected.sort_values("dataset").reset_index(drop=True)


def select_by_leave_one_dataset_validation_mean_minus_std(
    dataset_summary: pd.DataFrame,
    *,
    std_penalty: float,
) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for dataset in sorted(test["dataset"].dropna().astype(str).unique()):
        selector_pool = validation[~validation["dataset"].astype(str).eq(dataset)]
        if selector_pool.empty:
            selector_pool = validation
        target_test = test[test["dataset"].astype(str).eq(dataset)]
        eligible = _runs_with_dataset_coverage(
            selector_pool,
            required_datasets=selector_pool["dataset"].dropna().astype(str).unique(),
        )
        eligible &= set(target_test["run_id"].astype(str).unique())
        selector_pool = selector_pool[selector_pool["run_id"].isin(eligible)]
        target_test = target_test[target_test["run_id"].isin(eligible)]
        if selector_pool.empty or target_test.empty:
            continue
        scores = (
            selector_pool.groupby("run_id", dropna=False)
            .agg(
                validation_mean_iou=("mean_iou", "mean"),
                validation_std_iou=("mean_iou", lambda values: values.std(ddof=0)),
                validation_min_iou=("mean_iou", "min"),
                validation_mean_f1=("mean_f1", "mean"),
            )
            .reset_index()
        )
        scores["validation_stability_score"] = (
            scores["validation_mean_iou"] - std_penalty * scores["validation_std_iou"]
        )
        selected_run = (
            scores.sort_values(
                [
                    "validation_stability_score",
                    "validation_mean_iou",
                    "validation_mean_f1",
                    "run_id",
                ],
                ascending=[False, False, False, True],
            )
            .head(1)
        )
        selected = target_test.merge(selected_run, on="run_id", how="inner")
        if selected.empty:
            continue
        selected["validation_std_penalty"] = std_penalty
        selected["selection_rule"] = (
            f"leave_one_dataset_mean_minus_{_margin_token(std_penalty)}std_selected"
        )
        parts.append(selected)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("dataset").reset_index(drop=True)


def select_by_leave_one_dataset_validation(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for dataset in sorted(test["dataset"].dropna().astype(str).unique()):
        selector_pool = validation[~validation["dataset"].astype(str).eq(dataset)]
        if selector_pool.empty:
            selector_pool = validation
        eligible = _runs_with_dataset_coverage(
            selector_pool,
            required_datasets=selector_pool["dataset"].dropna().astype(str).unique(),
        )
        target_test = test[test["dataset"].astype(str).eq(dataset)]
        eligible &= set(target_test["run_id"].astype(str).unique())
        selector_pool = selector_pool[selector_pool["run_id"].isin(eligible)]
        target_test = target_test[target_test["run_id"].isin(eligible)]
        if selector_pool.empty or target_test.empty:
            continue
        selected_run = (
            selector_pool.groupby("run_id", dropna=False)
            .agg(validation_mean_iou=("mean_iou", "mean"), validation_mean_f1=("mean_f1", "mean"))
            .reset_index()
            .sort_values(["validation_mean_iou", "validation_mean_f1", "run_id"], ascending=[False, False, True])
            .head(1)[["run_id", "validation_mean_iou"]]
        )
        selected = target_test.merge(selected_run, on="run_id", how="inner")
        parts.append(selected)
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True)
    rows["selection_rule"] = "leave_one_dataset_validation_selected"
    return rows.sort_values("dataset").reset_index(drop=True)


def guarded_margin_outputs(
    dataset_summary: pd.DataFrame,
    *,
    global_selected: pd.DataFrame,
    margins: Sequence[float],
) -> dict[str, pd.DataFrame]:
    if global_selected.empty or "run_id" not in global_selected.columns:
        return {}
    baseline_run_ids = sorted(global_selected["run_id"].astype(str).unique())
    if len(baseline_run_ids) != 1:
        return {}
    baseline_run_id = baseline_run_ids[0]
    return {
        f"guarded_validation_margin_{_margin_token(margin)}": select_by_guarded_margin(
            dataset_summary,
            baseline_run_id=baseline_run_id,
            validation_margin=margin,
        )
        for margin in margins
    }


def select_by_guarded_margin(
    dataset_summary: pd.DataFrame,
    *,
    baseline_run_id: str,
    validation_margin: float,
) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for dataset in sorted(test["dataset"].dropna().astype(str).unique()):
        validation_dataset = validation[validation["dataset"].astype(str).eq(dataset)]
        test_dataset = test[test["dataset"].astype(str).eq(dataset)]
        baseline_validation = validation_dataset[
            validation_dataset["run_id"].astype(str).eq(baseline_run_id)
        ]
        baseline_test = test_dataset[test_dataset["run_id"].astype(str).eq(baseline_run_id)]
        if baseline_validation.empty or baseline_test.empty:
            continue
        baseline_mean = float(baseline_validation["mean_iou"].iloc[0])
        best = (
            validation_dataset.sort_values(
                ["mean_iou", "mean_f1", "run_id"],
                ascending=[False, False, True],
            )
            .head(1)
        )
        if best.empty:
            continue
        best_row = best.iloc[0]
        best_gain = float(best_row["mean_iou"]) - baseline_mean
        selected_run_id = (
            str(best_row["run_id"])
            if best_gain >= validation_margin
            else baseline_run_id
        )
        selected = test_dataset[test_dataset["run_id"].astype(str).eq(selected_run_id)].copy()
        if selected.empty:
            continue
        selected["baseline_run_id"] = baseline_run_id
        selected["baseline_validation_mean_iou"] = baseline_mean
        selected["best_validation_run_id"] = str(best_row["run_id"])
        selected["best_validation_mean_iou"] = float(best_row["mean_iou"])
        selected["validation_gain_over_baseline"] = best_gain
        selected["validation_margin"] = validation_margin
        selected["selection_rule"] = f"guarded_validation_margin_{_margin_token(validation_margin)}"
        parts.append(selected)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("dataset").reset_index(drop=True)


def summarize_selected_outputs(selected: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for rule, frame in selected.items():
        if frame.empty:
            continue
        records.append(
            {
                "selection_rule": rule,
                "mean_iou": float(frame["mean_iou"].mean()),
                "mean_f1": float(frame["mean_f1"].mean()),
                "n_datasets": int(frame["dataset"].nunique()),
                "n_folds": int(frame["n_folds"].sum()),
                "n_cells": int(frame["n_cells"].sum()),
                "selected_run_ids": ";".join(sorted(frame["run_id"].astype(str).unique())),
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values("mean_iou", ascending=False)
        .reset_index(drop=True)
    )


def write_markdown_report(
    *,
    output_path: Path,
    fixed: pd.DataFrame,
    selected_overall: pd.DataFrame,
    validation_selected: pd.DataFrame,
    global_selected: pd.DataFrame,
    stability_selected: dict[str, pd.DataFrame],
    leave_one_dataset_selected: pd.DataFrame,
    guarded_selected: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# QA-Pruned ENSEMBLE Validation Selection",
        "",
        "Strict table scope: QA and ENSEMBLE rows only. No SIMPLE rows are selected here.",
        "",
        "## Selection Rules",
        "",
    ]
    lines.extend(_markdown_table(selected_overall))
    lines.extend(["", "## Fixed Runs", ""])
    lines.extend(_markdown_table(fixed.head(30)))
    lines.extend(["", "## Validation Selected By Dataset", ""])
    lines.extend(_markdown_table(_selection_columns(validation_selected)))
    lines.extend(["", "## Global Validation Selected", ""])
    lines.extend(_markdown_table(_selection_columns(global_selected)))
    if stability_selected:
        stability_frames = [frame for frame in stability_selected.values() if not frame.empty]
        if stability_frames:
            stability = pd.concat(stability_frames, ignore_index=True)
            lines.extend(["", "## Stability Validation Selected", ""])
            lines.extend(_markdown_table(_selection_columns(stability)))
    lines.extend(["", "## Leave-One-Dataset Validation Selected", ""])
    lines.extend(_markdown_table(_selection_columns(leave_one_dataset_selected)))
    if guarded_selected:
        guarded = pd.concat(guarded_selected.values(), ignore_index=True)
        lines.extend(["", "## Guarded Validation Selected", ""])
        lines.extend(_markdown_table(_selection_columns(guarded)))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "selection_rule",
        "method_label",
        "ensemble_variant",
        "qa_pruning_policy",
        "soft_gate",
        "threshold",
        "validation_mean_iou",
        "baseline_validation_mean_iou",
        "validation_gain_over_baseline",
        "validation_margin",
        "validation_std_iou",
        "validation_min_iou",
        "validation_stability_score",
        "validation_std_penalty",
        "mean_iou",
        "mean_f1",
    ]
    present = [column for column in columns if column in frame.columns]
    return frame[present] if present else frame


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


def _runs_with_dataset_coverage(
    frame: pd.DataFrame,
    *,
    required_datasets: Sequence[str],
) -> set[str]:
    required = {str(dataset) for dataset in required_datasets}
    if not required:
        return set()
    covered: set[str] = set()
    for run_id, group in frame.groupby("run_id", dropna=False):
        datasets = set(group["dataset"].dropna().astype(str))
        if required.issubset(datasets):
            covered.add(str(run_id))
    return covered


def _full_coverage_validation_test(
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = _runs_with_dataset_coverage(
        validation,
        required_datasets=validation["dataset"].dropna().astype(str).unique(),
    )
    eligible &= _runs_with_dataset_coverage(
        test,
        required_datasets=test["dataset"].dropna().astype(str).unique(),
    )
    return validation[validation["run_id"].isin(eligible)], test[test["run_id"].isin(eligible)]


def _margin_token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p").replace("-", "m")


def _run_id(row: pd.Series) -> str:
    parts = [
        str(row.get("result_family", "")),
        str(row.get("method_key", "")),
        str(row.get("ensemble_variant", "")),
        str(row.get("qa_pruning_policy", "")),
        str(row.get("soft_gate", "")),
        str(row.get("weights_column", "")),
        str(row.get("boundary_support_column", "")),
        _float_token(row.get("boundary_floor", "")),
        str(row.get("boundary_floor_column", "")),
        str(row.get("boundary_iterations", "")),
        f"thr{float(row.get('threshold', 0.0)):.3f}",
    ]
    return "__".join(parts)


def _float_token(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return str(value)
    return f"{float(numeric):.6f}"


if __name__ == "__main__":
    main()
