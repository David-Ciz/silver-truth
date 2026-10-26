#!/usr/bin/env python3
"""Build selection tables for internal object-vote ENSEMBLE probes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUNS = (
    (
        "uniform_thr050",
        Path(
            "hpc_results/soft_hybrid_probe/"
            "qa_weighted_object_vote_probe_all_available_full_uniform_thr050"
        ),
    ),
    (
        "boundary_full_floor025",
        Path(
            "hpc_results/soft_hybrid_probe/"
            "boundary_object_vote_probe_all_available_full_boundary_floor025"
        ),
    ),
    (
        "boundary_full_floor075",
        Path(
            "hpc_results/soft_hybrid_probe/"
            "boundary_object_vote_probe_all_available_full_boundary_floor075"
        ),
    ),
    (
        "boundary_full_floor090",
        Path(
            "hpc_results/soft_hybrid_probe/"
            "boundary_object_vote_probe_all_available_full_boundary_floor090"
        ),
    ),
)

MODALITY_RULE_RUNS = {
    "BF": "uniform_thr050",
    "DIC": "boundary_full_floor075",
    "Fluo": "boundary_full_floor025",
}


@dataclass(frozen=True)
class ObjectVoteRun:
    run_id: str
    root: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("RUN_ID", "ROOT"),
        help=(
            "Object-vote run root. Can be repeated. Defaults to the current "
            "uniform and boundary-aware all-available probes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpc_results/soft_hybrid_probe/object_vote_selection_tables"),
    )
    parser.add_argument(
        "--boundary-margin-baseline-run-id",
        default=None,
        help=(
            "Optional baseline run for boundary-margin selection, e.g. set_thr055. "
            "When set, boundary runs are selected only if their validation gain "
            "over this baseline clears --boundary-validation-margin."
        ),
    )
    parser.add_argument(
        "--boundary-run-pattern",
        default="boundary",
        help="Substring identifying boundary-aware run IDs for margin selection.",
    )
    parser.add_argument(
        "--boundary-validation-margin",
        action="append",
        type=float,
        default=None,
        help="Validation IoU margin for boundary-aware run selection. Can be repeated.",
    )
    args = parser.parse_args()

    runs = (
        [ObjectVoteRun(run_id, Path(root)) for run_id, root in args.run]
        if args.run
        else [ObjectVoteRun(run_id, root) for run_id, root in DEFAULT_RUNS]
    )
    rows = collect_summary_rows(runs)
    if rows.empty:
        raise SystemExit("No weighted_object_vote_summary.csv rows found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dataset = summarize_by_dataset(rows)
    validation_selected = select_by_validation(candidate_dataset)
    global_validation_selected = select_by_global_validation(candidate_dataset)
    leave_one_dataset_selected = select_by_leave_one_dataset_validation(
        candidate_dataset
    )
    modality_rule = select_by_modality_rule(candidate_dataset)
    selected_outputs = {
        "validation_selected_by_dataset": validation_selected,
        "global_validation_selected": global_validation_selected,
        "leave_one_dataset_validation_selected": leave_one_dataset_selected,
        "modality_rule": modality_rule,
    }
    boundary_margin_selected = _boundary_margin_outputs(
        candidate_dataset,
        baseline_run_id=args.boundary_margin_baseline_run_id,
        boundary_run_pattern=args.boundary_run_pattern,
        margins=tuple(args.boundary_validation_margin or ()),
    )
    selected_outputs.update(boundary_margin_selected)
    leave_one_dataset_boundary_margin_selected = (
        select_by_leave_one_dataset_boundary_margin(
            candidate_dataset,
            baseline_run_id=args.boundary_margin_baseline_run_id,
            boundary_run_pattern=args.boundary_run_pattern,
            validation_margins=tuple(args.boundary_validation_margin or ()),
        )
        if args.boundary_margin_baseline_run_id and args.boundary_validation_margin
        else pd.DataFrame()
    )
    if not leave_one_dataset_boundary_margin_selected.empty:
        selected_outputs["leave_one_dataset_boundary_margin_selected"] = (
            leave_one_dataset_boundary_margin_selected
        )
    overall = summarize_selected_outputs(selected_outputs, fixed_runs=candidate_dataset)

    rows.to_csv(args.output_dir / "object_vote_candidate_fold_rows.csv", index=False)
    candidate_dataset.to_csv(
        args.output_dir / "object_vote_candidate_dataset_summary.csv",
        index=False,
    )
    validation_selected.to_csv(
        args.output_dir / "object_vote_validation_selected_dataset_summary.csv",
        index=False,
    )
    global_validation_selected.to_csv(
        args.output_dir / "object_vote_global_validation_selected_dataset_summary.csv",
        index=False,
    )
    leave_one_dataset_selected.to_csv(
        (
            args.output_dir
            / "object_vote_leave_one_dataset_validation_selected_dataset_summary.csv"
        ),
        index=False,
    )
    if boundary_margin_selected:
        pd.concat(boundary_margin_selected.values(), ignore_index=True).to_csv(
            args.output_dir / "object_vote_boundary_margin_selected_dataset_summary.csv",
            index=False,
        )
    if not leave_one_dataset_boundary_margin_selected.empty:
        leave_one_dataset_boundary_margin_selected.to_csv(
            (
                args.output_dir
                / "object_vote_leave_one_dataset_boundary_margin_selected_dataset_summary.csv"
            ),
            index=False,
        )
    modality_rule.to_csv(
        args.output_dir / "object_vote_modality_rule_dataset_summary.csv",
        index=False,
    )
    overall.to_csv(args.output_dir / "object_vote_selection_overall_summary.csv", index=False)
    write_markdown_report(
        output_path=args.output_dir / "object_vote_selection_summary.md",
        candidate_dataset=candidate_dataset,
        validation_selected=validation_selected,
        global_validation_selected=global_validation_selected,
        leave_one_dataset_selected=leave_one_dataset_selected,
        boundary_margin_selected=boundary_margin_selected,
        leave_one_dataset_boundary_margin_selected=(
            leave_one_dataset_boundary_margin_selected
        ),
        modality_rule=modality_rule,
        overall=overall,
    )

    print(f"Wrote object-vote selection tables to {args.output_dir}")


def collect_summary_rows(runs: list[ObjectVoteRun]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        for path in sorted(run.root.glob("**/weighted_object_vote_summary.csv")):
            frame = pd.read_csv(path)
            frame["run_id"] = run.run_id
            frame["run_root"] = str(run.root)
            frame["summary_path"] = str(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    rows["mean_iou"] = pd.to_numeric(rows["mean_iou"], errors="coerce")
    rows["mean_f1"] = pd.to_numeric(rows["mean_f1"], errors="coerce")
    return rows


def summarize_by_dataset(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "run_id",
        "dataset",
        "crop_tag",
        "variant",
        "split",
        "row_family",
        "method_key",
        "method_label",
        "weights_column",
        "boundary_support_column",
        "boundary_floor",
        "boundary_iterations",
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


def select_by_validation(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    selected_runs = (
        validation.sort_values(["dataset", "mean_iou", "run_id"], ascending=[True, False, True])
        .groupby("dataset", dropna=False)
        .head(1)[["dataset", "run_id", "mean_iou"]]
        .rename(columns={"mean_iou": "validation_mean_iou"})
    )
    test = dataset_summary[dataset_summary["split"].eq("test")]
    selected = test.merge(selected_runs, on=["dataset", "run_id"], how="inner")
    selected["selection_rule"] = "validation_selected_by_dataset"
    return selected.sort_values("dataset").reset_index(drop=True)


def select_by_global_validation(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    if validation.empty:
        return pd.DataFrame()
    selected_run = (
        validation.groupby("run_id", dropna=False)["mean_iou"]
        .mean()
        .reset_index(name="validation_mean_iou")
        .sort_values(["validation_mean_iou", "run_id"], ascending=[False, True])
        .head(1)
    )
    test = dataset_summary[dataset_summary["split"].eq("test")]
    selected = test.merge(selected_run, on="run_id", how="inner")
    selected["selection_rule"] = "global_validation_selected"
    return selected.sort_values("dataset").reset_index(drop=True)


def select_by_leave_one_dataset_validation(
    dataset_summary: pd.DataFrame,
) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty:
        return pd.DataFrame()

    selected_parts: list[pd.DataFrame] = []
    for dataset in sorted(test["dataset"].dropna().astype(str).unique()):
        selector_pool = validation[~validation["dataset"].astype(str).eq(dataset)]
        if selector_pool.empty:
            selector_pool = validation
        selected_run = (
            selector_pool.groupby("run_id", dropna=False)["mean_iou"]
            .mean()
            .reset_index(name="validation_mean_iou")
            .sort_values(["validation_mean_iou", "run_id"], ascending=[False, True])
            .head(1)
        )
        selected = test[test["dataset"].astype(str).eq(dataset)].merge(
            selected_run, on="run_id", how="inner"
        )
        selected_parts.append(selected)
    if not selected_parts:
        return pd.DataFrame()
    rows = pd.concat(selected_parts, ignore_index=True)
    rows["selection_rule"] = "leave_one_dataset_validation_selected"
    return rows.sort_values("dataset").reset_index(drop=True)


def select_by_boundary_margin(
    dataset_summary: pd.DataFrame,
    *,
    baseline_run_id: str,
    boundary_run_pattern: str = "boundary",
    validation_margin: float = 0.0,
) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty:
        return pd.DataFrame()

    selected_runs = _boundary_margin_selected_runs(
        validation,
        baseline_run_id=baseline_run_id,
        boundary_run_pattern=boundary_run_pattern,
        validation_margin=validation_margin,
    )

    selected = test.merge(
        selected_runs[
            [
                "dataset",
                "run_id",
                "validation_mean_iou",
                "validation_best_run_id",
                "validation_best_mean_iou",
                "baseline_run_id",
                "baseline_validation_mean_iou",
            ]
        ],
        on=["dataset", "run_id"],
        how="inner",
    )
    selected["selection_rule"] = _boundary_margin_rule_name(validation_margin)
    selected["boundary_validation_margin"] = float(validation_margin)
    return selected.sort_values("dataset").reset_index(drop=True)


def select_by_leave_one_dataset_boundary_margin(
    dataset_summary: pd.DataFrame,
    *,
    baseline_run_id: str,
    boundary_run_pattern: str = "boundary",
    validation_margins: tuple[float, ...],
) -> pd.DataFrame:
    validation = dataset_summary[dataset_summary["split"].eq("validation")]
    test = dataset_summary[dataset_summary["split"].eq("test")]
    if validation.empty or test.empty or not validation_margins:
        return pd.DataFrame()

    selected_parts: list[pd.DataFrame] = []
    for dataset in sorted(test["dataset"].dropna().astype(str).unique()):
        selector_pool = validation[~validation["dataset"].astype(str).eq(dataset)]
        if selector_pool.empty:
            selector_pool = validation
        selected_margin = _select_boundary_margin_from_validation_pool(
            selector_pool,
            baseline_run_id=baseline_run_id,
            boundary_run_pattern=boundary_run_pattern,
            validation_margins=validation_margins,
        )
        if selected_margin is None:
            continue

        heldout_validation = validation[validation["dataset"].astype(str).eq(dataset)]
        selected_runs = _boundary_margin_selected_runs(
            heldout_validation,
            baseline_run_id=baseline_run_id,
            boundary_run_pattern=boundary_run_pattern,
            validation_margin=selected_margin["boundary_validation_margin"],
        )
        selected = test[test["dataset"].astype(str).eq(dataset)].merge(
            selected_runs,
            on=["dataset", "run_id"],
            how="inner",
        )
        if selected.empty:
            continue
        selected["selection_rule"] = "leave_one_dataset_boundary_margin_selected"
        selected["boundary_validation_margin"] = float(
            selected_margin["boundary_validation_margin"]
        )
        selected["margin_selection_validation_mean_iou"] = float(
            selected_margin["margin_selection_validation_mean_iou"]
        )
        selected_parts.append(selected)

    if not selected_parts:
        return pd.DataFrame()
    return (
        pd.concat(selected_parts, ignore_index=True)
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def _boundary_margin_selected_runs(
    validation: pd.DataFrame,
    *,
    baseline_run_id: str,
    boundary_run_pattern: str,
    validation_margin: float,
) -> pd.DataFrame:
    best = (
        validation.sort_values(
            ["dataset", "mean_iou", "run_id"], ascending=[True, False, True]
        )
        .groupby("dataset", dropna=False)
        .head(1)[["dataset", "run_id", "mean_iou"]]
        .rename(
            columns={
                "run_id": "validation_best_run_id",
                "mean_iou": "validation_best_mean_iou",
            }
        )
    )
    baseline = validation[validation["run_id"].eq(baseline_run_id)][
        ["dataset", "run_id", "mean_iou"]
    ].rename(
        columns={
            "run_id": "baseline_run_id",
            "mean_iou": "baseline_validation_mean_iou",
        }
    )
    selected_runs = best.merge(baseline, on="dataset", how="left")
    is_boundary = selected_runs["validation_best_run_id"].astype(str).str.contains(
        boundary_run_pattern, regex=False, na=False
    )
    clears_margin = (
        selected_runs["validation_best_mean_iou"]
        - selected_runs["baseline_validation_mean_iou"]
    ) >= float(validation_margin)
    selected_runs["run_id"] = selected_runs["validation_best_run_id"]
    selected_runs.loc[is_boundary & ~clears_margin, "run_id"] = selected_runs.loc[
        is_boundary & ~clears_margin, "baseline_run_id"
    ]
    selected_validation = validation[["dataset", "run_id", "mean_iou"]].rename(
        columns={"mean_iou": "validation_mean_iou"}
    )
    return selected_runs.merge(
        selected_validation,
        on=["dataset", "run_id"],
        how="left",
    )


def _select_boundary_margin_from_validation_pool(
    validation: pd.DataFrame,
    *,
    baseline_run_id: str,
    boundary_run_pattern: str,
    validation_margins: tuple[float, ...],
) -> dict[str, float] | None:
    records: list[dict[str, float]] = []
    for margin in validation_margins:
        selected_runs = _boundary_margin_selected_runs(
            validation,
            baseline_run_id=baseline_run_id,
            boundary_run_pattern=boundary_run_pattern,
            validation_margin=margin,
        )
        if selected_runs.empty:
            continue
        records.append(
            {
                "boundary_validation_margin": float(margin),
                "margin_selection_validation_mean_iou": float(
                    selected_runs["validation_mean_iou"].mean()
                ),
            }
        )
    if not records:
        return None
    selected = (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["margin_selection_validation_mean_iou", "boundary_validation_margin"],
            ascending=[False, False],
        )
        .iloc[0]
    )
    return {
        "boundary_validation_margin": float(selected["boundary_validation_margin"]),
        "margin_selection_validation_mean_iou": float(
            selected["margin_selection_validation_mean_iou"]
        ),
    }


def select_by_modality_rule(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    test = dataset_summary[dataset_summary["split"].eq("test")].copy()
    test["modality_prefix"] = test["dataset"].astype(str).str.split("-", n=1).str[0]
    test["selected_run_id"] = test["modality_prefix"].map(MODALITY_RULE_RUNS)
    selected = test[test["run_id"].eq(test["selected_run_id"])].copy()
    selected["selection_rule"] = "modality_rule"
    return selected.drop(columns=["modality_prefix", "selected_run_id"]).sort_values(
        "dataset"
    )


def summarize_selected_outputs(
    selected_outputs: dict[str, pd.DataFrame],
    *,
    fixed_runs: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    fixed_test = fixed_runs[fixed_runs["split"].eq("test")]
    for run_id, group in fixed_test.groupby("run_id", dropna=False):
        records.append(_overall_record(f"fixed::{run_id}", group))
    for rule, group in selected_outputs.items():
        if not group.empty:
            records.append(_overall_record(rule, group))
    return pd.DataFrame.from_records(records).sort_values("mean_iou", ascending=False)


def _boundary_margin_outputs(
    dataset_summary: pd.DataFrame,
    *,
    baseline_run_id: str | None,
    boundary_run_pattern: str,
    margins: tuple[float, ...],
) -> dict[str, pd.DataFrame]:
    if baseline_run_id is None or not margins:
        return {}
    outputs = {}
    for margin in margins:
        rule = _boundary_margin_rule_name(margin)
        outputs[rule] = select_by_boundary_margin(
            dataset_summary,
            baseline_run_id=baseline_run_id,
            boundary_run_pattern=boundary_run_pattern,
            validation_margin=margin,
        )
    return outputs


def _boundary_margin_rule_name(margin: float) -> str:
    encoded = f"{float(margin):.4f}".replace(".", "p").rstrip("0").rstrip("p")
    return f"boundary_margin_selected_{encoded}"


def _overall_record(selection_rule: str, rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "selection_rule": selection_rule,
        "mean_iou": float(rows["mean_iou"].mean()),
        "mean_f1": float(rows["mean_f1"].mean()),
        "datasets": int(rows["dataset"].nunique()),
        "selected_runs": ";".join(
            f"{row['dataset']}={row['run_id']}" for row in rows.to_dict("records")
        ),
    }


def write_markdown_report(
    *,
    output_path: Path,
    candidate_dataset: pd.DataFrame,
    validation_selected: pd.DataFrame,
    global_validation_selected: pd.DataFrame,
    leave_one_dataset_selected: pd.DataFrame,
    boundary_margin_selected: dict[str, pd.DataFrame],
    leave_one_dataset_boundary_margin_selected: pd.DataFrame,
    modality_rule: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    lines = [
        "# Object Vote Selection Summary",
        "",
        "## Overall",
        "",
        overall.to_markdown(index=False),
        "",
        "## Validation-Selected By Dataset",
        "",
        validation_selected[
            ["dataset", "run_id", "validation_mean_iou", "mean_iou", "mean_f1"]
        ].to_markdown(index=False),
        "",
        "## Global Validation-Selected",
        "",
        global_validation_selected[
            ["dataset", "run_id", "validation_mean_iou", "mean_iou", "mean_f1"]
        ].to_markdown(index=False),
        "",
        "## Leave-One-Dataset Validation-Selected",
        "",
        leave_one_dataset_selected[
            ["dataset", "run_id", "validation_mean_iou", "mean_iou", "mean_f1"]
        ].to_markdown(index=False),
        "",
        "## Boundary-Margin Selected",
        "",
        _boundary_margin_markdown(boundary_margin_selected),
        "",
        "## Leave-One-Dataset Boundary-Margin Selected",
        "",
        _leave_one_dataset_boundary_margin_markdown(
            leave_one_dataset_boundary_margin_selected
        ),
        "",
        "## Modality Rule",
        "",
        modality_rule[["dataset", "run_id", "mean_iou", "mean_f1"]].to_markdown(
            index=False
        ),
        "",
        "## Candidate Test Rows",
        "",
        candidate_dataset[candidate_dataset["split"].eq("test")][
            ["dataset", "run_id", "mean_iou", "mean_f1"]
        ].to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _boundary_margin_markdown(boundary_margin_selected: dict[str, pd.DataFrame]) -> str:
    if not boundary_margin_selected:
        return "No boundary-margin selectors requested."
    rows = pd.concat(boundary_margin_selected.values(), ignore_index=True)
    return rows[
        [
            "selection_rule",
            "dataset",
            "run_id",
            "validation_mean_iou",
            "validation_best_run_id",
            "validation_best_mean_iou",
            "baseline_validation_mean_iou",
            "mean_iou",
            "mean_f1",
        ]
    ].to_markdown(index=False)


def _leave_one_dataset_boundary_margin_markdown(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No leave-one-dataset boundary-margin selector requested."
    return rows[
        [
            "dataset",
            "run_id",
            "boundary_validation_margin",
            "margin_selection_validation_mean_iou",
            "validation_mean_iou",
            "mean_iou",
            "mean_f1",
        ]
    ].to_markdown(index=False)


if __name__ == "__main__":
    main()
