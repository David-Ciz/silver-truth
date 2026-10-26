#!/usr/bin/env python3
"""Build validation-selected QA-pruned SIMPLE policy tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["dataset", "crop_tag", "variant", "fold"]
DATASET_COLUMNS = ["dataset", "crop_tag", "variant"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-rows",
        type=Path,
        default=Path(
            "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
            "all_available_simple_policy_comparison_rows.csv"
        ),
    )
    parser.add_argument(
        "--paper-summary",
        type=Path,
        default=Path("hpc_results/combined/reports/paper_result_summary_all_available.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
            "validation_policy_selection"
        ),
    )
    args = parser.parse_args()

    rows = attach_validation_metrics(pd.read_csv(args.comparison_rows))
    paper = pd.read_csv(args.paper_summary)
    context = _baseline_context(paper)

    fold_selected = select_by_fold_validation(rows)
    dataset_selected = select_by_dataset_validation(rows)
    fixed = select_fixed_policy(rows, policy="source_balanced_topk")

    selected = pd.concat(
        [
            _with_selection_rule(fixed, "fixed_source_balanced_topk"),
            _with_selection_rule(fold_selected, "validation_selected_by_fold"),
            _with_selection_rule(dataset_selected, "validation_selected_by_dataset"),
        ],
        ignore_index=True,
    )
    selected = selected.merge(context, on=KEY_COLUMNS, how="left")
    selected["delta_vs_fusion"] = selected["mean_iou"] - selected["best_fusion_iou"]
    selected["delta_vs_competitor"] = (
        selected["mean_iou"] - selected["best_competitor_iou"]
    )

    policy_table = summarize_policy_table(rows)
    dataset_summary = summarize_dataset_selection(selected)
    overall = summarize_overall_selection(dataset_summary)
    markdown = render_selection_markdown(dataset_summary, overall)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "policy_fold_rows_with_validation.csv", index=False)
    policy_table.to_csv(args.output_dir / "policy_overall_summary.csv", index=False)
    selected.to_csv(args.output_dir / "selected_policy_fold_rows.csv", index=False)
    dataset_summary.to_csv(
        args.output_dir / "selected_policy_dataset_summary.csv", index=False
    )
    overall.to_csv(args.output_dir / "selected_policy_overall_summary.csv", index=False)
    (args.output_dir / "selected_policy_summary.md").write_text(markdown)

    print(f"Wrote validation-selection tables to {args.output_dir}")
    print(overall.to_string(index=False))


def attach_validation_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    if {"filtered_candidate_rows", "input_candidate_rows"}.issubset(rows.columns):
        rows["retained_fraction"] = (
            pd.to_numeric(rows["filtered_candidate_rows"], errors="coerce")
            / pd.to_numeric(rows["input_candidate_rows"], errors="coerce")
        )
    records = []
    for row in rows.to_dict("records"):
        path = row.get("path")
        if not isinstance(path, str) or not path:
            records.append(_empty_validation_record(row))
            continue
        leaderboard_path = Path(path).parent / "fusion" / "fusion_crops_leaderboard.csv"
        if not leaderboard_path.exists():
            records.append(_empty_validation_record(row))
            continue
        leaderboard = pd.read_csv(leaderboard_path)
        model = str(row.get("fusion_backend", "")).upper()
        model_rows = leaderboard[leaderboard["model"].astype(str).str.upper().eq(model)]
        if model_rows.empty:
            model_rows = leaderboard.head(1)
        leader = model_rows.iloc[0]
        records.append(
            {
                **row,
                "validation_mean_iou": leader.get("eval_validation_mean_jaccard"),
                "validation_mean_f1": leader.get("eval_validation_mean_f1"),
                "validation_count": leader.get("eval_validation_count"),
                "train_mean_iou": leader.get("eval_train_mean_jaccard"),
                "test_mean_iou": leader.get("eval_test_mean_jaccard", row.get("mean_iou")),
                "test_mean_f1": leader.get("eval_test_mean_f1", row.get("mean_f1")),
                "leaderboard_path": str(leaderboard_path),
            }
        )
    return pd.DataFrame.from_records(records)


def select_by_fold_validation(rows: pd.DataFrame) -> pd.DataFrame:
    sortable = rows.dropna(subset=["validation_mean_iou"]).copy()
    return (
        sortable.sort_values(
            KEY_COLUMNS + ["validation_mean_iou", "policy"],
            ascending=[True, True, True, True, False, True],
        )
        .groupby(KEY_COLUMNS, dropna=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def select_by_dataset_validation(rows: pd.DataFrame) -> pd.DataFrame:
    sortable = rows.dropna(subset=["validation_mean_iou"]).copy()
    policy_scores = (
        sortable.groupby(DATASET_COLUMNS + ["policy"], dropna=False)
        .agg(validation_policy_mean_iou=("validation_mean_iou", "mean"))
        .reset_index()
        .sort_values(
            DATASET_COLUMNS + ["validation_policy_mean_iou", "policy"],
            ascending=[True, True, True, False, True],
        )
    )
    winners = (
        policy_scores.groupby(DATASET_COLUMNS, dropna=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected = sortable.merge(
        winners[DATASET_COLUMNS + ["policy", "validation_policy_mean_iou"]],
        on=DATASET_COLUMNS + ["policy"],
        how="inner",
    )
    return selected.reset_index(drop=True)


def select_fixed_policy(rows: pd.DataFrame, *, policy: str) -> pd.DataFrame:
    return rows[rows["policy"].astype(str).eq(policy)].copy().reset_index(drop=True)


def summarize_policy_table(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["policy", "fusion_backend"], dropna=False)
        .agg(
            folds=("fold", "count"),
            mean_validation_iou=("validation_mean_iou", "mean"),
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            mean_retained_fraction=("retained_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["mean_validation_iou", "mean_iou"], ascending=[False, False])
    )


def summarize_dataset_selection(selected: pd.DataFrame) -> pd.DataFrame:
    return (
        selected.groupby(["selection_rule"] + DATASET_COLUMNS, dropna=False)
        .agg(
            n_folds=("fold", "count"),
            selected_policy=("policy", _join_unique),
            mean_validation_iou=("validation_mean_iou", "mean"),
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            best_fusion_iou=("best_fusion_iou", "mean"),
            best_competitor_iou=("best_competitor_iou", "mean"),
            silver_truth_iou=("silver_truth_iou", "mean"),
            delta_vs_fusion=("delta_vs_fusion", "mean"),
            delta_vs_competitor=("delta_vs_competitor", "mean"),
        )
        .reset_index()
        .sort_values(["selection_rule", "dataset"])
    )


def summarize_overall_selection(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        dataset_summary.groupby("selection_rule", dropna=False)
        .agg(
            n_datasets=("dataset", "count"),
            mean_validation_iou=("mean_validation_iou", "mean"),
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            best_fusion_iou=("best_fusion_iou", "mean"),
            best_competitor_iou=("best_competitor_iou", "mean"),
            silver_truth_iou=("silver_truth_iou", "mean"),
            delta_vs_fusion=("delta_vs_fusion", "mean"),
            delta_vs_competitor=("delta_vs_competitor", "mean"),
        )
        .reset_index()
        .sort_values("mean_iou", ascending=False)
    )


def render_selection_markdown(
    dataset_summary: pd.DataFrame, overall: pd.DataFrame
) -> str:
    display_dataset = dataset_summary.copy()
    display_overall = overall.copy()
    numeric_columns = [
        "mean_validation_iou",
        "mean_iou",
        "best_fusion_iou",
        "best_competitor_iou",
        "silver_truth_iou",
        "delta_vs_fusion",
        "delta_vs_competitor",
    ]
    for frame in (display_dataset, display_overall):
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = frame[column].map(lambda value: f"{value:.4f}")
    return "\n".join(
        [
            "# QA-Pruned SIMPLE Validation Selection",
            "",
            "## Overall",
            "",
            display_overall.to_markdown(index=False),
            "",
            "## Dataset Summary",
            "",
            display_dataset.to_markdown(index=False),
            "",
        ]
    )


def _baseline_context(paper: pd.DataFrame) -> pd.DataFrame:
    fold_rows = paper[paper["row_type"].astype(str).eq("fold")].copy()
    columns = KEY_COLUMNS + [
        "best_fusion_iou",
        "best_competitor_iou",
        "silver_truth_iou",
    ]
    return fold_rows[columns]


def _with_selection_rule(rows: pd.DataFrame, rule: str) -> pd.DataFrame:
    rows = rows.copy()
    rows["selection_rule"] = rule
    return rows


def _empty_validation_record(row: dict[str, object]) -> dict[str, object]:
    return {
        **row,
        "validation_mean_iou": pd.NA,
        "validation_mean_f1": pd.NA,
        "validation_count": pd.NA,
        "train_mean_iou": pd.NA,
        "test_mean_iou": row.get("mean_iou"),
        "test_mean_f1": row.get("mean_f1"),
        "leaderboard_path": "",
    }


def _join_unique(values: pd.Series) -> str:
    return ";".join(sorted({str(value) for value in values.dropna()}))


if __name__ == "__main__":
    main()
