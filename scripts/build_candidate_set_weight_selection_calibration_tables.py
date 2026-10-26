#!/usr/bin/env python3
"""Build candidate-set ENSEMBLE selection/calibration summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ROOT = Path("hpc_results/soft_hybrid_probe")
DEFAULT_OUTPUT_DIR = (
    DEFAULT_ROOT / "candidate_set_weight_selection_calibration"
)


MAIN_STRICT_IDS = {
    "candidate_set_weight_mlp_trainonly",
    "candidate_set_weight_ridge_trainonly",
    "candidate_set_weight_mlp_lodo",
    "candidate_set_weight_mlp_lodo_no_dataset_fixed_thr055",
    "candidate_set_weight_mlp_lodo_no_dataset_boundary_margin",
}

SELECTION_CONTROLS = [
    {
        "control_id": "mlp_trainonly_all_available",
        "method_label": "Candidate-set MLP train-only selector diagnostics",
        "rel_dir": "candidate_set_weight_object_vote_selection_trainonly_all_available",
        "include_swept_margin": False,
    },
    {
        "control_id": "ridge_trainonly_all_available",
        "method_label": "Candidate-set ridge train-only selector diagnostics",
        "rel_dir": "candidate_set_weight_object_vote_selection_ridge_trainonly_all_available",
        "include_swept_margin": False,
    },
    {
        "control_id": "mlp_trainonly_lodo",
        "method_label": "Candidate-set MLP LODO selector diagnostics",
        "rel_dir": "candidate_set_weight_object_vote_selection_lodo",
        "include_swept_margin": False,
    },
    {
        "control_id": "mlp_trainonly_lodo_no_dataset",
        "method_label": "Candidate-set MLP LODO no-dataset selector diagnostics",
        "rel_dir": "candidate_set_weight_object_vote_selection_lodo_no_dataset",
        "include_swept_margin": True,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    build_tables(root=args.root, output_dir=args.output_dir)


def build_tables(*, root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = build_overall_table(root)
    dataset = build_dataset_table(root)
    overall.to_csv(
        output_dir / "candidate_set_weight_selection_calibration_overall.csv",
        index=False,
    )
    dataset.to_csv(
        output_dir / "candidate_set_weight_selection_calibration_dataset.csv",
        index=False,
    )
    _write_markdown(
        output_dir / "candidate_set_weight_selection_calibration_summary.md",
        overall,
        dataset,
    )


def build_overall_table(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(_main_strict_overall_rows(root))
    for control in SELECTION_CONTROLS:
        rows.extend(_selection_overall_rows(root, **control))
    rows.extend(_evolution_context_overall_rows(root))
    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table
    table["sort_group"] = table["comparison_group"].map(
        {
            "strict_qa_ensemble": 0,
            "selector_diagnostic": 1,
            "legacy_qa_ensemble_context": 2,
            "deployable_context": 3,
            "external_context": 4,
            "fusion_baseline_context": 5,
        }
    ).fillna(9)
    return (
        table.sort_values(
            ["sort_group", "mean_iou", "method_id"],
            ascending=[True, False, True],
        )
        .drop(columns=["sort_group"])
        .reset_index(drop=True)
    )


def build_dataset_table(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(_main_strict_dataset_rows(root))
    for control in SELECTION_CONTROLS:
        rows.extend(_selection_dataset_rows(root, **control))
    rows.extend(_evolution_context_dataset_rows(root))
    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table
    table["sort_group"] = table["comparison_group"].map(
        {
            "strict_qa_ensemble": 0,
            "selector_diagnostic": 1,
            "legacy_qa_ensemble_context": 2,
            "deployable_context": 3,
            "external_context": 4,
            "fusion_baseline_context": 5,
        }
    ).fillna(9)
    return (
        table.sort_values(
            ["dataset", "sort_group", "mean_iou", "method_id"],
            ascending=[True, True, False, True],
        )
        .drop(columns=["sort_group"])
        .reset_index(drop=True)
    )


def _main_strict_overall_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "phase2_ablation_matrix" / "phase2_strict_qa_ensemble_overall_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df[df["ablation_id"].isin(MAIN_STRICT_IDS)].copy()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            _overall_record(
                comparison_group="strict_qa_ensemble",
                method_id=row["ablation_id"],
                method_label=row["method_label"],
                row_family=row["row_family"],
                selection_rule=row["evidence_kind"],
                datasets=row["datasets"],
                mean_iou=row["mean_iou"],
                mean_f1=row.get("mean_f1"),
                mean_validation_iou=row.get("mean_validation_iou"),
                notes=row.get("notes", ""),
                source_path=str(path),
            )
        )
    return rows


def _main_strict_dataset_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "phase2_ablation_matrix" / "phase2_strict_qa_ensemble_dataset_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df[df["ablation_id"].isin(MAIN_STRICT_IDS)].copy()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            _dataset_record(
                comparison_group="strict_qa_ensemble",
                method_id=row["ablation_id"],
                method_label=row["method_label"],
                row_family=row["row_family"],
                dataset=row["dataset"],
                selection_rule=row["evidence_kind"],
                mean_iou=row["mean_iou"],
                mean_f1=row.get("mean_f1"),
                mean_validation_iou=row.get("mean_validation_iou"),
                selected_method=row.get("selected_methods", ""),
                notes=row.get("notes", ""),
                source_path=str(path),
            )
        )
    return rows


def _selection_overall_rows(
    root: Path,
    *,
    control_id: str,
    method_label: str,
    rel_dir: str,
    include_swept_margin: bool,
) -> list[dict[str, Any]]:
    path = root / rel_dir / "object_vote_selection_overall_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rules = {
        "validation_selected_by_dataset",
        "leave_one_dataset_boundary_margin_selected",
        "fixed::set_thr055",
    }
    if include_swept_margin:
        rules.add("boundary_margin_selected_0p015")
    df = df[df["selection_rule"].isin(rules)].copy()
    rows = []
    for _, row in df.iterrows():
        rule = str(row["selection_rule"])
        rows.append(
            _overall_record(
                comparison_group="selector_diagnostic",
                method_id=f"{control_id}::{rule}",
                method_label=method_label,
                row_family="ensemble_qa",
                selection_rule=rule,
                datasets=row["datasets"],
                mean_iou=row["mean_iou"],
                mean_f1=row.get("mean_f1"),
                mean_validation_iou=pd.NA,
                notes=_selector_note(control_id, rule),
                source_path=str(path),
            )
        )
    return rows


def _selection_dataset_rows(
    root: Path,
    *,
    control_id: str,
    method_label: str,
    rel_dir: str,
    include_swept_margin: bool,
) -> list[dict[str, Any]]:
    selection_root = root / rel_dir
    rows: list[dict[str, Any]] = []
    rows.extend(
        _selection_dataset_file_rows(
            selection_root / "object_vote_validation_selected_dataset_summary.csv",
            control_id=control_id,
            method_label=method_label,
            selection_rule="validation_selected_by_dataset",
        )
    )
    rows.extend(
        _selection_dataset_file_rows(
            selection_root
            / "object_vote_leave_one_dataset_boundary_margin_selected_dataset_summary.csv",
            control_id=control_id,
            method_label=method_label,
            selection_rule="leave_one_dataset_boundary_margin_selected",
        )
    )
    rows.extend(
        _fixed_run_dataset_rows(
            selection_root / "object_vote_candidate_dataset_summary.csv",
            control_id=control_id,
            method_label=method_label,
            run_id="set_thr055",
        )
    )
    if include_swept_margin:
        rows.extend(
            _selection_dataset_file_rows(
                selection_root / "object_vote_boundary_margin_selected_dataset_summary.csv",
                control_id=control_id,
                method_label=method_label,
                selection_rule="boundary_margin_selected_0p015",
            )
        )
    return rows


def _selection_dataset_file_rows(
    path: Path,
    *,
    control_id: str,
    method_label: str,
    selection_rule: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "selection_rule" in df.columns:
        df = df[df["selection_rule"].eq(selection_rule)].copy()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            _dataset_record(
                comparison_group="selector_diagnostic",
                method_id=f"{control_id}::{selection_rule}",
                method_label=method_label,
                row_family="ensemble_qa",
                dataset=row["dataset"],
                selection_rule=selection_rule,
                mean_iou=row["mean_iou"],
                mean_f1=row.get("mean_f1"),
                mean_validation_iou=row.get("validation_mean_iou", pd.NA),
                selected_method=row.get("run_id", ""),
                notes=_selector_note(control_id, selection_rule),
                source_path=str(path),
            )
        )
    return rows


def _fixed_run_dataset_rows(
    path: Path,
    *,
    control_id: str,
    method_label: str,
    run_id: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df[df["split"].eq("test") & df["run_id"].eq(run_id)].copy()
    rows = []
    selection_rule = f"fixed::{run_id}"
    for _, row in df.iterrows():
        rows.append(
            _dataset_record(
                comparison_group="selector_diagnostic",
                method_id=f"{control_id}::{selection_rule}",
                method_label=method_label,
                row_family="ensemble_qa",
                dataset=row["dataset"],
                selection_rule=selection_rule,
                mean_iou=row["mean_iou"],
                mean_f1=row.get("mean_f1"),
                mean_validation_iou=pd.NA,
                selected_method=row.get("run_id", ""),
                notes=_selector_note(control_id, selection_rule),
                source_path=str(path),
            )
        )
    return rows


def _evolution_context_overall_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "qa_ensemble_evolution" / "qa_ensemble_evolution_overall_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    keep = {
        "legacy_simple_nn_validation_selected",
        "legacy_simple_nn_foldwise_best",
        "full_pipeline_best_policy",
        "competitor",
        "silver_truth",
        "fusion_only_best",
    }
    df = df[df["method_id"].isin(keep)].copy()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            _overall_record(
                comparison_group=_context_group(str(row["method_id"])),
                method_id=row["method_id"],
                method_label=row["method_label"],
                row_family=row["row_family"],
                selection_rule=row.get("table_scope", ""),
                datasets=row["datasets"],
                mean_iou=row["mean_iou"],
                mean_f1=row.get("mean_f1"),
                mean_validation_iou=row.get("mean_validation_iou"),
                notes=row.get("notes", ""),
                source_path=str(path),
            )
        )
    return rows


def _evolution_context_dataset_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "qa_ensemble_evolution" / "qa_ensemble_evolution_dataset_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    keep = {
        "legacy_simple_nn_validation_selected",
        "legacy_simple_nn_foldwise_best",
        "full_pipeline_best_policy",
        "competitor",
        "silver_truth",
        "fusion_only_best",
    }
    df = df[df["method_id"].isin(keep)].copy()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            _dataset_record(
                comparison_group=_context_group(str(row["method_id"])),
                method_id=row["method_id"],
                method_label=row["method_label"],
                row_family=row["row_family"],
                dataset=row["dataset"],
                selection_rule=row.get("table_scope", ""),
                mean_iou=row["mean_iou"],
                mean_f1=row.get("mean_f1"),
                mean_validation_iou=row.get("mean_validation_iou"),
                selected_method=row.get("selected_methods", ""),
                notes=row.get("notes", ""),
                source_path=str(path),
            )
        )
    return rows


def _context_group(method_id: str) -> str:
    if method_id.startswith("legacy_simple_nn"):
        return "legacy_qa_ensemble_context"
    if method_id == "full_pipeline_best_policy":
        return "deployable_context"
    if method_id == "fusion_only_best":
        return "fusion_baseline_context"
    return "external_context"


def _selector_note(control_id: str, selection_rule: str) -> str:
    if selection_rule == "leave_one_dataset_boundary_margin_selected":
        return (
            "Boundary margin selected from validation rows on other datasets; "
            "no held-out test leakage."
        )
    if selection_rule.startswith("boundary_margin_selected"):
        return (
            "Swept boundary-margin diagnostic; useful for calibration analysis, "
            "not a non-test-selected headline result."
        )
    if selection_rule.startswith("fixed::"):
        return "Fixed candidate object-vote variant used as robustness baseline."
    if "lodo_no_dataset" in control_id:
        return "No-dataset LODO validation selector diagnostic."
    return "Validation-selected candidate-set object-vote selector diagnostic."


def _overall_record(
    *,
    comparison_group: str,
    method_id: str,
    method_label: str,
    row_family: str,
    selection_rule: str,
    datasets: Any,
    mean_iou: Any,
    mean_f1: Any,
    mean_validation_iou: Any,
    notes: Any,
    source_path: str,
) -> dict[str, Any]:
    return {
        "comparison_group": comparison_group,
        "method_id": method_id,
        "method_label": method_label,
        "row_family": row_family,
        "selection_rule": selection_rule,
        "datasets": datasets,
        "mean_iou": mean_iou,
        "mean_f1": mean_f1,
        "mean_validation_iou": mean_validation_iou,
        "notes": notes,
        "source_path": source_path,
    }


def _dataset_record(
    *,
    comparison_group: str,
    method_id: str,
    method_label: str,
    row_family: str,
    dataset: str,
    selection_rule: str,
    mean_iou: Any,
    mean_f1: Any,
    mean_validation_iou: Any,
    selected_method: Any,
    notes: Any,
    source_path: str,
) -> dict[str, Any]:
    return {
        "comparison_group": comparison_group,
        "method_id": method_id,
        "method_label": method_label,
        "row_family": row_family,
        "dataset": dataset,
        "selection_rule": selection_rule,
        "mean_iou": mean_iou,
        "mean_f1": mean_f1,
        "mean_validation_iou": mean_validation_iou,
        "selected_method": selected_method,
        "notes": notes,
        "source_path": source_path,
    }


def _write_markdown(path: Path, overall: pd.DataFrame, dataset: pd.DataFrame) -> None:
    lines = [
        "# Candidate-Set ENSEMBLE Selection Calibration",
        "",
        (
            "Strict rows are QA/ENSEMBLE methods. Deployable rows and external "
            "baselines are context. FUSION-only rows are external baselines, not "
            "our ENSEMBLE method."
        ),
        "",
        "## Overall",
        "",
    ]
    display_cols = [
        "comparison_group",
        "method_id",
        "selection_rule",
        "datasets",
        "mean_iou",
        "mean_f1",
    ]
    lines.append(
        _format_markdown_values(overall[display_cols]).to_markdown(
            index=False,
            floatfmt=".6f",
        )
    )
    lines.extend(
        [
            "",
            "## Main Takeaways",
            "",
            "- Fixed `set_thr055` remains the robust no-dataset LODO row.",
            (
                "- Honest leave-one-dataset boundary-margin selection ties the "
                "main all-dataset MLP/ridge validation-selected rows but does "
                "not improve them."
            ),
            (
                "- Honest boundary-margin selection underperforms both fixed "
                "`set_thr055` and validation selection in LODO settings."
            ),
            (
                "- Test-swept boundary-margin rows should stay diagnostic; they "
                "explain calibration behavior but are not headline method rows."
            ),
            "",
            "## Dataset Detail",
            "",
        ]
    )
    detail_cols = [
        "comparison_group",
        "dataset",
        "method_id",
        "selection_rule",
        "mean_iou",
        "mean_f1",
        "selected_method",
    ]
    lines.append(
        _format_markdown_values(dataset[detail_cols]).to_markdown(
            index=False,
            floatfmt=".6f",
        )
    )
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
