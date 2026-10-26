#!/usr/bin/env python3
"""Build QA/ENSEMBLE evolution tables from completed result summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PHASE2_DIR = Path("hpc_results/soft_hybrid_probe/phase2_ablation_matrix")
DEFAULT_LEGACY_SUMMARY = Path(
    "hpc_results/combined/reports/paper_result_summary_all_available_qa_ensemble_only.csv"
)
DEFAULT_CONTEXT_SUMMARY = Path(
    "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
    "all_available_simple_policy_overall_context_summary.csv"
)
DEFAULT_CONTEXT_DATASET = Path(
    "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
    "all_available_simple_policy_context_dataset_selected.csv"
)
DEFAULT_BADMASS_SELECTION_DIR = Path(
    "hpc_results/soft_hybrid_probe/"
    "candidate_set_weight_deep_sets_badmass_validation_selection"
)

BADMASS_SELECTION_ROWS = {
    "validation_selected_by_dataset": {
        "method_id": "candidate_set_weight_deep_sets_badmass_validation_selected_by_dataset",
        "method_label": (
            "Bad-mass/Deep Sets object vote, validation-selected by dataset"
        ),
        "comparison_group": "strict_fold_protocol",
        "notes": (
            "Strict internal ENSEMBLE follow-up; selected per dataset using "
            "validation split IoU."
        ),
    },
    "global_validation_selected": {
        "method_id": "candidate_set_weight_deep_sets_badmass_global_validation_selected",
        "method_label": "Deep Sets object vote, global validation-selected",
        "comparison_group": "strict_fold_protocol",
        "notes": (
            "Strict internal ENSEMBLE follow-up; one run selected by global "
            "validation IoU."
        ),
    },
    "leave_one_dataset_validation_selected": {
        "method_id": (
            "candidate_set_weight_deep_sets_badmass_leave_one_dataset_"
            "validation_selected"
        ),
        "method_label": (
            "Bad-mass/Deep Sets object vote, leave-one-dataset validation-selected"
        ),
        "comparison_group": "strict_cross_dataset_stress",
        "notes": (
            "Strict internal ENSEMBLE leave-one-dataset selection stress test; "
            "do not use as the fold-protocol headline."
        ),
    },
    "fixed::badmass_w010_topk5_thr055": {
        "method_id": "candidate_set_weight_deep_sets_badmass_w010_topk5_thr055_fixed",
        "method_label": "Bad-mass Deep Sets top-5 object vote, fixed w=0.10 thr=0.55",
        "comparison_group": "strict_fold_protocol",
        "notes": (
            "Strict internal ENSEMBLE fixed-policy follow-up from the bad-mass "
            "Deep Sets sweep."
        ),
    },
}

BADMASS_DATASET_SUMMARY_FILES = {
    "validation_selected_by_dataset": "object_vote_validation_selected_dataset_summary.csv",
    "global_validation_selected": "object_vote_global_validation_selected_dataset_summary.csv",
    "leave_one_dataset_validation_selected": (
        "object_vote_leave_one_dataset_validation_selected_dataset_summary.csv"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    parser.add_argument("--legacy-summary", type=Path, default=DEFAULT_LEGACY_SUMMARY)
    parser.add_argument("--context-summary", type=Path, default=DEFAULT_CONTEXT_SUMMARY)
    parser.add_argument("--context-dataset", type=Path, default=DEFAULT_CONTEXT_DATASET)
    parser.add_argument(
        "--badmass-selection-dir",
        type=Path,
        default=DEFAULT_BADMASS_SELECTION_DIR,
        help="Directory containing bad-mass Deep Sets validation-selection summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpc_results/soft_hybrid_probe/qa_ensemble_evolution"),
    )
    args = parser.parse_args()

    build_tables(
        phase2_dir=args.phase2_dir,
        legacy_summary=args.legacy_summary,
        context_summary=args.context_summary,
        context_dataset=args.context_dataset,
        badmass_selection_dir=args.badmass_selection_dir,
        output_dir=args.output_dir,
    )


def build_tables(
    *,
    phase2_dir: Path,
    legacy_summary: Path,
    context_summary: Path,
    context_dataset: Path,
    output_dir: Path,
    badmass_selection_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = build_overall_table(
        phase2_dir=phase2_dir,
        legacy_summary=legacy_summary,
        context_summary=context_summary,
        badmass_selection_dir=badmass_selection_dir,
    )
    dataset = build_dataset_table(
        phase2_dir=phase2_dir,
        legacy_summary=legacy_summary,
        context_dataset=context_dataset,
        badmass_selection_dir=badmass_selection_dir,
    )
    overall.to_csv(output_dir / "qa_ensemble_evolution_overall_summary.csv", index=False)
    dataset.to_csv(output_dir / "qa_ensemble_evolution_dataset_summary.csv", index=False)
    _write_markdown(output_dir / "qa_ensemble_evolution_summary.md", overall, dataset)


def build_overall_table(
    *,
    phase2_dir: Path,
    legacy_summary: Path,
    context_summary: Path,
    badmass_selection_dir: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_strict_overall_rows(phase2_dir))
    rows.extend(_badmass_selection_overall_rows(badmass_selection_dir))
    rows.extend(_legacy_overall_rows(legacy_summary))
    rows.extend(_context_overall_rows(context_summary))
    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table
    table["sort_group"] = table["comparison_group"].map(
        {
            "strict_fold_protocol": 0,
            "strict_cross_dataset_stress": 1,
            "legacy_simple_nn": 2,
            "deployable_context": 3,
            "external_context": 4,
        }
    ).fillna(9)
    table = table.sort_values(
        ["sort_group", "mean_iou", "method_id"], ascending=[True, False, True]
    )
    return table.drop(columns=["sort_group"]).reset_index(drop=True)


def build_dataset_table(
    *,
    phase2_dir: Path,
    legacy_summary: Path,
    context_dataset: Path,
    badmass_selection_dir: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_strict_dataset_rows(phase2_dir))
    rows.extend(_badmass_selection_dataset_rows(badmass_selection_dir))
    rows.extend(_legacy_dataset_rows(legacy_summary))
    rows.extend(_context_dataset_rows(context_dataset))
    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table
    table["sort_group"] = table["comparison_group"].map(
        {
            "strict_fold_protocol": 0,
            "strict_cross_dataset_stress": 1,
            "legacy_simple_nn": 2,
            "deployable_context": 3,
            "external_context": 4,
        }
    ).fillna(9)
    table = table.sort_values(
        ["dataset", "sort_group", "mean_iou", "method_id"],
        ascending=[True, True, False, True],
    )
    return table.drop(columns=["sort_group"]).reset_index(drop=True)


def _strict_overall_rows(phase2_dir: Path) -> list[dict[str, object]]:
    path = phase2_dir / "phase2_strict_qa_ensemble_overall_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "comparison_group": _strict_comparison_group(row),
                "method_id": row["ablation_id"],
                "method_label": row["method_label"],
                "row_family": row["row_family"],
                "table_scope": row["table_scope"],
                "datasets": row["datasets"],
                "mean_iou": row["mean_iou"],
                "mean_f1": row.get("mean_f1"),
                "mean_validation_iou": row.get("mean_validation_iou"),
                "notes": row.get("notes", ""),
            }
        )
    return rows


def _strict_dataset_rows(phase2_dir: Path) -> list[dict[str, object]]:
    path = phase2_dir / "phase2_strict_qa_ensemble_dataset_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "comparison_group": _strict_comparison_group(row),
                "method_id": row["ablation_id"],
                "method_label": row["method_label"],
                "row_family": row["row_family"],
                "table_scope": row["table_scope"],
                "dataset": row["dataset"],
                "mean_iou": row["mean_iou"],
                "mean_f1": row.get("mean_f1"),
                "mean_validation_iou": row.get("mean_validation_iou"),
                "selected_methods": row.get("selected_methods", ""),
                "notes": row.get("notes", ""),
            }
        )
    return rows


def _strict_comparison_group(row: pd.Series) -> str:
    tokens = " ".join(
        str(row.get(column, ""))
        for column in ("ablation_id", "phase2_role", "evidence_kind", "notes")
    ).lower()
    if "lodo" in tokens or "cross_dataset" in tokens or "leave-one-dataset" in tokens:
        return "strict_cross_dataset_stress"
    return "strict_fold_protocol"


def _badmass_selection_overall_rows(
    badmass_selection_dir: Path | None,
) -> list[dict[str, object]]:
    if badmass_selection_dir is None:
        return []
    path = badmass_selection_dir / "object_vote_selection_overall_summary.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    rows = []
    for _, row in df[df["selection_rule"].isin(BADMASS_SELECTION_ROWS)].iterrows():
        metadata = BADMASS_SELECTION_ROWS[str(row["selection_rule"])]
        rows.append(
            {
                "comparison_group": metadata["comparison_group"],
                "method_id": metadata["method_id"],
                "method_label": metadata["method_label"],
                "row_family": "ensemble_qa",
                "table_scope": "strict_qa_ensemble",
                "datasets": row["datasets"],
                "mean_iou": row["mean_iou"],
                "mean_f1": row.get("mean_f1"),
                "mean_validation_iou": pd.NA,
                "notes": metadata["notes"],
            }
        )
    return rows


def _badmass_selection_dataset_rows(
    badmass_selection_dir: Path | None,
) -> list[dict[str, object]]:
    if badmass_selection_dir is None:
        return []

    rows: list[dict[str, object]] = []
    for selection_rule, filename in BADMASS_DATASET_SUMMARY_FILES.items():
        path = badmass_selection_dir / filename
        if not path.exists():
            continue
        rows.extend(_badmass_selected_dataset_rows(path, selection_rule))

    rows.extend(_badmass_fixed_dataset_rows(badmass_selection_dir))
    return rows


def _badmass_selected_dataset_rows(
    path: Path,
    selection_rule: str,
) -> list[dict[str, object]]:
    metadata = BADMASS_SELECTION_ROWS[selection_rule]
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "comparison_group": metadata["comparison_group"],
                "method_id": metadata["method_id"],
                "method_label": metadata["method_label"],
                "row_family": "ensemble_qa",
                "table_scope": "strict_qa_ensemble",
                "dataset": row["dataset"],
                "mean_iou": row["mean_iou"],
                "mean_f1": row.get("mean_f1"),
                "mean_validation_iou": row.get("validation_mean_iou"),
                "selected_methods": row.get("run_id", ""),
                "notes": metadata["notes"],
            }
        )
    return rows


def _badmass_fixed_dataset_rows(badmass_selection_dir: Path) -> list[dict[str, object]]:
    path = badmass_selection_dir / "object_vote_candidate_dataset_summary.csv"
    if not path.exists():
        return []

    selection_rule = "fixed::badmass_w010_topk5_thr055"
    metadata = BADMASS_SELECTION_ROWS[selection_rule]
    df = pd.read_csv(path)
    run_id = "badmass_w010_topk5_thr055"
    test_rows = df[df["run_id"].eq(run_id) & df["split"].eq("test")].copy()
    validation_rows = df[
        df["run_id"].eq(run_id) & df["split"].eq("validation")
    ].copy()
    validation_by_dataset = validation_rows.set_index("dataset")["mean_iou"].to_dict()

    rows = []
    for _, row in test_rows.iterrows():
        rows.append(
            {
                "comparison_group": metadata["comparison_group"],
                "method_id": metadata["method_id"],
                "method_label": metadata["method_label"],
                "row_family": "ensemble_qa",
                "table_scope": "strict_qa_ensemble",
                "dataset": row["dataset"],
                "mean_iou": row["mean_iou"],
                "mean_f1": row.get("mean_f1"),
                "mean_validation_iou": validation_by_dataset.get(row["dataset"], pd.NA),
                "selected_methods": run_id,
                "notes": metadata["notes"],
            }
        )
    return rows


def _legacy_overall_rows(legacy_summary: Path) -> list[dict[str, object]]:
    if not legacy_summary.exists():
        return []
    mean_rows = pd.read_csv(legacy_summary)
    mean_rows = mean_rows[mean_rows["row_type"].eq("mean")].copy()
    return [
        {
            "comparison_group": "legacy_simple_nn",
            "method_id": "legacy_simple_nn_validation_selected",
            "method_label": "Legacy simple-NN QA/ENSEMBLE, validation-selected",
            "row_family": "legacy_qa_ensemble",
            "table_scope": "legacy_context",
            "datasets": int(mean_rows["dataset"].nunique()),
            "mean_iou": float(mean_rows["validation_selected_ours_iou"].mean()),
            "mean_f1": pd.NA,
            "mean_validation_iou": float(
                mean_rows["validation_selected_validation_iou"].mean()
            ),
            "notes": "Previous paper QA/ENSEMBLE-only validation-selected result.",
        },
        {
            "comparison_group": "legacy_simple_nn",
            "method_id": "legacy_simple_nn_foldwise_best",
            "method_label": "Legacy simple-NN QA/ENSEMBLE, foldwise-best",
            "row_family": "legacy_qa_ensemble",
            "table_scope": "legacy_context",
            "datasets": int(mean_rows["dataset"].nunique()),
            "mean_iou": float(mean_rows["best_ours_iou"].mean()),
            "mean_f1": pd.NA,
            "mean_validation_iou": pd.NA,
            "notes": "Previous paper QA/ENSEMBLE-only foldwise-best diagnostic.",
        },
    ]


def _legacy_dataset_rows(legacy_summary: Path) -> list[dict[str, object]]:
    if not legacy_summary.exists():
        return []
    mean_rows = pd.read_csv(legacy_summary)
    mean_rows = mean_rows[mean_rows["row_type"].eq("mean")].copy()
    rows = []
    for _, row in mean_rows.iterrows():
        rows.append(
            {
                "comparison_group": "legacy_simple_nn",
                "method_id": "legacy_simple_nn_validation_selected",
                "method_label": "Legacy simple-NN QA/ENSEMBLE, validation-selected",
                "row_family": "legacy_qa_ensemble",
                "table_scope": "legacy_context",
                "dataset": row["dataset"],
                "mean_iou": row["validation_selected_ours_iou"],
                "mean_f1": pd.NA,
                "mean_validation_iou": row["validation_selected_validation_iou"],
                "selected_methods": row["validation_selected_ours_method"],
                "notes": "Previous paper QA/ENSEMBLE-only validation-selected result.",
            }
        )
        rows.append(
            {
                "comparison_group": "legacy_simple_nn",
                "method_id": "legacy_simple_nn_foldwise_best",
                "method_label": "Legacy simple-NN QA/ENSEMBLE, foldwise-best",
                "row_family": "legacy_qa_ensemble",
                "table_scope": "legacy_context",
                "dataset": row["dataset"],
                "mean_iou": row["best_ours_iou"],
                "mean_f1": pd.NA,
                "mean_validation_iou": pd.NA,
                "selected_methods": row["best_ours_method"],
                "notes": "Previous paper QA/ENSEMBLE-only foldwise-best diagnostic.",
            }
        )
    return rows


def _context_overall_rows(context_summary: Path) -> list[dict[str, object]]:
    if not context_summary.exists():
        return []
    df = pd.read_csv(context_summary)
    keep = {
        "full_pipeline_best_policy": (
            "deployable_context",
            "QA-pruned SIMPLE deployable context",
        ),
        "fusion_only_best": ("external_context", "Best fusion-only baseline"),
        "competitor": ("external_context", "Best external competitor"),
        "silver_truth": ("external_context", "Silver truth"),
    }
    rows = []
    for _, row in df[df["row_family"].isin(keep)].iterrows():
        group, label = keep[row["row_family"]]
        rows.append(
            {
                "comparison_group": group,
                "method_id": row["row_family"],
                "method_label": label,
                "row_family": row["row_family"],
                "table_scope": "context_not_strict_qa_ensemble",
                "datasets": row["datasets"],
                "mean_iou": row["mean_iou"],
                "mean_f1": row.get("mean_f1"),
                "mean_validation_iou": pd.NA,
                "notes": _context_note(str(row["row_family"])),
            }
        )
    return rows


def _context_dataset_rows(context_dataset: Path) -> list[dict[str, object]]:
    if not context_dataset.exists():
        return []
    df = pd.read_csv(context_dataset)
    keep = {
        "full_pipeline_best_policy": (
            "deployable_context",
            "QA-pruned SIMPLE deployable context",
        ),
        "fusion_only_best": ("external_context", "Best fusion-only baseline"),
        "competitor": ("external_context", "Best external competitor"),
        "silver_truth": ("external_context", "Silver truth"),
    }
    rows = []
    df = df[df["fold"].eq("mean") & df["row_family"].isin(keep)].copy()
    for _, row in df.iterrows():
        group, label = keep[row["row_family"]]
        rows.append(
            {
                "comparison_group": group,
                "method_id": row["row_family"],
                "method_label": label,
                "row_family": row["row_family"],
                "table_scope": "context_not_strict_qa_ensemble",
                "dataset": row["dataset"],
                "mean_iou": row["mean_iou"],
                "mean_f1": row.get("mean_f1"),
                "mean_validation_iou": pd.NA,
                "selected_methods": row.get("method_key", ""),
                "notes": _context_note(str(row["row_family"])),
            }
        )
    return rows


def _context_note(row_family: str) -> str:
    if row_family == "full_pipeline_best_policy":
        return "Deployable QA-pruned SIMPLE context; not a strict ENSEMBLE row."
    if row_family == "fusion_only_best":
        return "External fusion baseline; FUSION is not ENSEMBLE and not our method."
    if row_family == "competitor":
        return "Best external competitor context."
    if row_family == "silver_truth":
        return "Official silver-truth target context."
    return "Context row."


def _write_markdown(path: Path, overall: pd.DataFrame, dataset: pd.DataFrame) -> None:
    display = overall[
        [
            "comparison_group",
            "method_label",
            "row_family",
            "datasets",
            "mean_iou",
            "mean_f1",
        ]
    ].copy()
    display = _format_markdown_values(display)
    lines = [
        "# QA/ENSEMBLE Evolution Summary",
        "",
        (
            "Strict fold-protocol rows are QA/ENSEMBLE methods evaluated under "
            "the standard fold protocol. Strict cross-dataset rows are LODO "
            "stress tests; do not use them to replace the fold-protocol "
            "headline. Deployable, fusion-only, competitor, and silver-truth "
            "rows are context only."
        ),
        "",
        "## Overall",
        "",
        display.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Dataset Detail",
        "",
    ]
    detail = dataset[
        [
            "comparison_group",
            "dataset",
            "method_label",
            "row_family",
            "mean_iou",
            "mean_f1",
        ]
    ].copy()
    detail = _format_markdown_values(detail)
    lines.append(detail.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_markdown_values(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.columns:
        if column.startswith("mean_"):
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    return formatted


if __name__ == "__main__":
    main()
