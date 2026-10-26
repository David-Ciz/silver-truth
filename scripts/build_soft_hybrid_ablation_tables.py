#!/usr/bin/env python
"""Build Phase 2 ablation tables from completed soft-hybrid runs.

The script intentionally keeps direct QA/ENSEMBLE candidate selection,
deployable pipeline context, and fusion baselines in separate row families.
See AGENTS.md for the method taxonomy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import nan
from pathlib import Path

import pandas as pd


ALL_DATASETS = {
    "BF-C2DL-HSC",
    "BF-C2DL-MuSC",
    "DIC-C2DH-HeLa",
    "Fluo-C2DL-MSC",
    "Fluo-N2DH-GOWT1",
}


@dataclass(frozen=True)
class StrategyRun:
    ablation_id: str
    method_label: str
    phase2_role: str
    coverage: str
    rel_dir: str
    notes: str
    row_family: str = "soft_hybrid_direct"
    table_scope: str = "deployable_candidate_selection"


STRATEGY_RUNS = [
    StrategyRun(
        ablation_id="pairwise_source",
        method_label="QA pairwise ranker + source prior",
        phase2_role="no_temporal_no_image_no_geometry_no_local_support",
        coverage="all_available",
        rel_dir="all_available_pairwise_source_trainonly_validation_selection",
        notes="All-dataset source-aware pairwise ranker before temporal/image/geometric/local-support extensions.",
    ),
    StrategyRun(
        ablation_id="no_qa",
        method_label="No QA score selector",
        phase2_role="no_qa",
        coverage="all_available",
        rel_dir="all_available_no_qa_fusion_temporal_morph_source_validation_selection",
        notes="All-dataset direct selector with QA weight set to zero; keeps fusion support, temporal, morphology, and source-prior terms.",
    ),
    StrategyRun(
        ablation_id="temporal_support",
        method_label="QA pairwise ranker + source prior + temporal support",
        phase2_role="no_image_no_geometry_no_local_support",
        coverage="all_available",
        rel_dir="all_available_temporal_support_pairwise_source_trainonly_validation_selection",
        notes="Adds temporal consistency features to the source-aware pairwise ranker.",
    ),
    StrategyRun(
        ablation_id="image_features",
        method_label="QA pairwise ranker + source prior + image features",
        phase2_role="image_support_candidate",
        coverage="all_available",
        rel_dir="all_available_image_features_pairwise_source_trainonly_validation_selection",
        notes="Adds learned/image-derived support features without the later rectangular/full-image variants.",
    ),
    StrategyRun(
        ablation_id="square_rect_image_features",
        method_label="QA pairwise ranker + rectangular image features",
        phase2_role="rectangular_image_support_candidate",
        coverage="all_available",
        rel_dir="all_available_square_rect_image_features_pairwise_source_trainonly_validation_selection",
        notes="Uses square/rectangular image-support features.",
    ),
    StrategyRun(
        ablation_id="full_image_features",
        method_label="QA pairwise ranker + full-image features",
        phase2_role="no_geometry_no_local_support",
        coverage="all_available",
        rel_dir="all_available_square_rect_full_image_features_pairwise_source_trainonly_validation_selection",
        notes="Full-image support without the explicit geometry feature extension.",
    ),
    StrategyRun(
        ablation_id="full_image_geometry_features",
        method_label="QA pairwise ranker + full-image + geometry features",
        phase2_role="no_local_support_boundary",
        coverage="all_available",
        rel_dir="all_available_square_rect_full_image_geometry_features_pairwise_source_trainonly_validation_selection",
        notes="Adds explicit geometry features to the full-image candidate selector.",
    ),
    StrategyRun(
        ablation_id="full_image_vs_geometry_combined",
        method_label="Validation-selected full-image/geometry candidate selector",
        phase2_role="combined_full_image_geometry_selection",
        coverage="all_available",
        rel_dir="all_available_full_image_vs_geometry_combined_validation_selection",
        notes="Validation selection across available full-image and geometry candidate-selector variants.",
    ),
    StrategyRun(
        ablation_id="full_image_vs_local_support_combined",
        method_label="Validation-selected full-image/local-support candidate selector",
        phase2_role="combined_full_image_local_support_selection",
        coverage="all_available",
        rel_dir="all_available_full_image_vs_local_support_combined_validation_selection",
        notes="Validation selection across full-image and local-support variants.",
    ),
    StrategyRun(
        ablation_id="full_image_vs_local_support_modes",
        method_label="Full available soft-hybrid selector with support modes",
        phase2_role="full_available_soft_hybrid_selector",
        coverage="all_available",
        rel_dir="all_available_full_image_vs_local_support_modes_combined_validation_selection",
        notes="Latest all-dataset support-mode candidate-selector run.",
    ),
    StrategyRun(
        ablation_id="no_source_prior_msc_gowt1",
        method_label="No source prior, temporal support ranker",
        phase2_role="no_source_prior",
        coverage="partial_msc_gowt1",
        rel_dir="msc_gowt1_temporal_support_pairwise_no_source_trainonly_validation_selection",
        notes="Partial coverage only; do not mix into all-dataset headline means.",
    ),
    StrategyRun(
        ablation_id="no_source_prior",
        method_label="No source prior, temporal support ranker",
        phase2_role="no_source_prior",
        coverage="all_available",
        rel_dir="all_available_temporal_support_pairwise_no_source_trainonly_validation_selection",
        notes="All-dataset no-source-prior run using temporal-support pairwise ranker features without source identity.",
    ),
    StrategyRun(
        ablation_id="source_blend0p5_msc_gowt1",
        method_label="Half-weight source prior, temporal support ranker",
        phase2_role="source_prior_sensitivity",
        coverage="partial_msc_gowt1",
        rel_dir="msc_gowt1_temporal_support_pairwise_source_blend0p5_trainonly_validation_selection",
        notes="Partial source-prior sensitivity run for MSC/GOWT1.",
    ),
]


REQUESTED_ABLATIONS = [
    {
        "requested_axis": "full model",
        "status": "available_proxy",
        "evidence": "full_image_vs_local_support_modes",
        "coverage": "all_available",
        "note": "Latest support-mode selector is the current full available candidate-selection proxy.",
    },
    {
        "requested_axis": "no QA",
        "status": "available",
        "evidence": "no_qa",
        "coverage": "all_available",
        "note": "All-dataset direct selector with QA weight set to zero is available.",
    },
    {
        "requested_axis": "no source prior",
        "status": "available",
        "evidence": "no_source_prior",
        "coverage": "all_available",
        "note": "All-dataset temporal-support pairwise run without source identity is available.",
    },
    {
        "requested_axis": "no temporal features",
        "status": "available_proxy",
        "evidence": "pairwise_source",
        "coverage": "all_available",
        "note": "Pairwise source run omits temporal support features.",
    },
    {
        "requested_axis": "no geometry",
        "status": "available_proxy",
        "evidence": "full_image_features",
        "coverage": "all_available",
        "note": "Full-image feature run omits the explicit geometry extension.",
    },
    {
        "requested_axis": "no image support",
        "status": "available_proxy",
        "evidence": "temporal_support",
        "coverage": "all_available",
        "note": "Temporal-support run omits image-support feature families.",
    },
    {
        "requested_axis": "no local-support/boundary",
        "status": "available_proxy",
        "evidence": "full_image_geometry_features",
        "coverage": "all_available",
        "note": "Full-image + geometry run omits the later local-support/boundary modes.",
    },
    {
        "requested_axis": "fusion-only",
        "status": "available_context",
        "evidence": "fusion_only_best",
        "coverage": "all_available",
        "note": "External fusion baseline context only; not our ENSEMBLE method.",
    },
    {
        "requested_axis": "oracle QA",
        "status": "diagnostic_only",
        "evidence": "msc oracle-gap files",
        "coverage": "partial_msc",
        "note": "Current oracle-gap files are MSC-only cluster-level diagnostics, not final reconstructed test IoU.",
    },
    {
        "requested_axis": "internal object-vote ENSEMBLE",
        "status": "available",
        "evidence": "object_vote_validation_selected",
        "coverage": "all_available",
        "note": "Validation-selected and modality-rule internal object-vote ENSEMBLE rows are available.",
    },
    {
        "requested_axis": "candidate-set learned ENSEMBLE",
        "status": "available",
        "evidence": "candidate_set_weight_mlp_leave_target_video_out_thr050",
        "coverage": "all_available",
        "note": (
            "Leave-target-video candidate-set weight model feeding internal "
            "object-vote ENSEMBLE is available. Single all-dataset models are "
            "diagnostic because they can see reciprocal target-video train rows."
        ),
    },
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _strategy_selected_rows(root: Path, run: StrategyRun) -> pd.DataFrame:
    path = root / run.rel_dir / "validation_selected_strategy.csv"
    df = _read_csv(path)
    df = df[df["fold"].astype(str).str.startswith("fold-")].copy()
    df["ablation_id"] = run.ablation_id
    df["method_label"] = run.method_label
    df["phase2_role"] = run.phase2_role
    df["coverage"] = run.coverage
    df["row_family"] = run.row_family
    df["table_scope"] = run.table_scope
    df["evidence_kind"] = "validation_selected_strategy"
    df["source_path"] = str(path)
    df["notes"] = run.notes
    df["selected_method"] = df["strategy"].astype(str) + ":" + df["preset"].astype(str)
    return df[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _strategy_foldwise_best_rows(root: Path, run: StrategyRun) -> pd.DataFrame:
    path = root / run.rel_dir / "strategy_candidates.csv"
    df = _read_csv(path)
    df = df[df["split"].eq("test")].copy()
    if df.empty:
        return pd.DataFrame()
    idx = df.groupby(["dataset", "fold"], sort=False)["mean_iou"].idxmax()
    best = df.loc[idx].copy()
    best["ablation_id"] = run.ablation_id
    best["method_label"] = run.method_label
    best["phase2_role"] = run.phase2_role
    best["coverage"] = run.coverage
    best["row_family"] = run.row_family
    best["table_scope"] = run.table_scope
    best["evidence_kind"] = "foldwise_best_diagnostic"
    best["source_path"] = str(path)
    best["notes"] = run.notes
    best["selected_method"] = best["strategy"].astype(str) + ":" + best["preset"].astype(str)
    best["selection_rule"] = "max_test_iou_within_ablation"
    best = best.rename(
        columns={
            "mean_iou": "test_mean_iou",
            "mean_f1": "test_mean_f1",
        }
    )
    best["validation_mean_iou"] = nan
    best["validation_mean_f1"] = nan
    return best[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _reporting_mode_selected_rows(root: Path) -> pd.DataFrame:
    path = root / "reporting_mode_selection_full_image_vs_local_support_modes_combined" / "validation_selected_reporting_mode.csv"
    df = _read_csv(path)
    df = df[df["row_type"].eq("fold")].copy()
    df["ablation_id"] = "reporting_mode_selected_full_image_local_support_modes"
    df["method_label"] = "Validation-selected reporting mode"
    df["phase2_role"] = "reporting_mode_selection"
    df["coverage"] = "all_available"
    df["row_family"] = "reporting_mode_selected"
    df["table_scope"] = "reporting_mode_context"
    df["evidence_kind"] = "validation_selected_reporting_mode"
    df["source_path"] = str(path)
    df["notes"] = "Validation selection across strict QA/ENSEMBLE, deployable, and soft-hybrid reporting modes."
    df["selection_rule"] = df.get("selection_rule", "max_validation_iou_across_reporting_modes")
    df = df.rename(
        columns={
            "selected_test_iou": "test_mean_iou",
            "selection_validation_iou": "validation_mean_iou",
        }
    )
    df["test_mean_f1"] = nan
    df["validation_mean_f1"] = nan
    return df[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _reporting_mode_foldwise_best_rows(root: Path) -> pd.DataFrame:
    path = root / "reporting_mode_selection_full_image_vs_local_support_modes_combined" / "reporting_mode_candidates.csv"
    df = _read_csv(path)
    df = df[df["row_type"].eq("fold")].copy()
    if df.empty:
        return pd.DataFrame()
    idx = df.groupby(["dataset", "fold"], sort=False)["selected_test_iou"].idxmax()
    best = df.loc[idx].copy()
    best["ablation_id"] = "reporting_mode_foldwise_best_full_image_local_support_modes"
    best["method_label"] = "Foldwise-best reporting mode"
    best["phase2_role"] = "reporting_mode_foldwise_oracle"
    best["coverage"] = "all_available"
    best["row_family"] = "reporting_mode_selected"
    best["table_scope"] = "reporting_mode_context"
    best["evidence_kind"] = "foldwise_best_diagnostic"
    best["source_path"] = str(path)
    best["notes"] = "Diagnostic upper bound from choosing the reporting mode with the best test IoU per fold."
    best["selection_rule"] = "max_test_iou_across_reporting_modes"
    best = best.rename(
        columns={
            "selected_test_iou": "test_mean_iou",
            "selection_validation_iou": "validation_mean_iou",
        }
    )
    best["test_mean_f1"] = nan
    best["validation_mean_f1"] = nan
    return best[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _object_vote_rows(
    root: Path,
    *,
    selection_rel_dir: str = "object_vote_selection_tables",
    selected_filename: str,
    ablation_id: str,
    method_label: str,
    phase2_role: str,
    evidence_kind: str,
    selection_rule: str,
    notes: str,
) -> pd.DataFrame:
    object_vote_root = root / selection_rel_dir
    selected_path = object_vote_root / selected_filename
    candidate_path = object_vote_root / "object_vote_candidate_fold_rows.csv"
    if not selected_path.exists() or not candidate_path.exists():
        return pd.DataFrame()

    selected = _read_csv(selected_path)
    candidate = _read_csv(candidate_path)
    candidate = candidate[candidate["split"].eq("test")].copy()
    selected_runs = selected[["dataset", "run_id"]].drop_duplicates()
    rows = candidate.merge(selected_runs, on=["dataset", "run_id"], how="inner")
    if rows.empty:
        return pd.DataFrame()

    validation_lookup = selected[["dataset", "run_id", "mean_iou", "mean_f1"]].rename(
        columns={
            "mean_iou": "selected_dataset_test_iou",
            "mean_f1": "selected_dataset_test_f1",
        }
    )
    if "validation_mean_iou" in selected.columns:
        validation_lookup = validation_lookup.merge(
            selected[["dataset", "run_id", "validation_mean_iou"]].drop_duplicates(),
            on=["dataset", "run_id"],
            how="left",
        )
    else:
        validation_lookup["validation_mean_iou"] = nan

    rows = rows.merge(validation_lookup, on=["dataset", "run_id"], how="left")
    rows["ablation_id"] = ablation_id
    rows["method_label"] = method_label
    rows["phase2_role"] = phase2_role
    rows["coverage"] = "all_available"
    rows["row_family"] = "ensemble_qa"
    rows["table_scope"] = "strict_qa_ensemble"
    rows["evidence_kind"] = evidence_kind
    rows["source_path"] = str(selected_path)
    rows["notes"] = notes
    rows["selected_method"] = rows["run_id"].astype(str)
    rows["selection_rule"] = selection_rule
    rows = rows.rename(
        columns={
            "mean_iou": "test_mean_iou",
            "mean_f1": "test_mean_f1",
        }
    )
    rows["validation_mean_f1"] = nan
    return rows[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _object_vote_fixed_run_rows(
    root: Path,
    *,
    selection_rel_dir: str,
    run_id: str,
    ablation_id: str,
    method_label: str,
    phase2_role: str,
    evidence_kind: str,
    row_family: str,
    notes: str,
) -> pd.DataFrame:
    object_vote_root = root / selection_rel_dir
    candidate_path = object_vote_root / "object_vote_candidate_fold_rows.csv"
    if not candidate_path.exists():
        return pd.DataFrame()
    candidate = _read_csv(candidate_path)
    rows = candidate[
        candidate["split"].eq("test") & candidate["run_id"].astype(str).eq(run_id)
    ].copy()
    if rows.empty:
        return pd.DataFrame()

    rows["ablation_id"] = ablation_id
    rows["method_label"] = method_label
    rows["phase2_role"] = phase2_role
    rows["coverage"] = "all_available"
    rows["row_family"] = row_family
    rows["table_scope"] = "strict_qa_ensemble"
    rows["evidence_kind"] = evidence_kind
    rows["source_path"] = str(candidate_path)
    rows["notes"] = notes
    rows["selected_method"] = rows["run_id"].astype(str)
    rows["selection_rule"] = f"fixed::{run_id}"
    rows = rows.rename(
        columns={
            "mean_iou": "test_mean_iou",
            "mean_f1": "test_mean_f1",
        }
    )
    rows["validation_mean_iou"] = nan
    rows["validation_mean_f1"] = nan
    return rows[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _object_vote_fixed_rows(root: Path) -> pd.DataFrame:
    object_vote_root = root / "object_vote_selection_tables"
    candidate_path = object_vote_root / "object_vote_candidate_fold_rows.csv"
    if not candidate_path.exists():
        return pd.DataFrame()
    candidate = _read_csv(candidate_path)
    rows = candidate[candidate["split"].eq("test")].copy()
    if rows.empty:
        return pd.DataFrame()

    run_metadata = {
        "uniform_thr050": {
            "ablation_id": "object_vote_uniform",
            "method_label": "Uniform object vote",
            "phase2_role": "internal_ensemble_uniform_vote",
            "row_family": "ensemble_only",
            "notes": "Internal full-pool uniform object-vote ENSEMBLE; no external fusion backend.",
        },
        "boundary_full_floor075": {
            "ablation_id": "object_vote_boundary_floor075",
            "method_label": "Boundary-aware object vote, floor 0.75",
            "phase2_role": "internal_ensemble_boundary_vote",
            "row_family": "ensemble_qa",
            "notes": "Fixed global boundary-aware internal object-vote ENSEMBLE using full-boundary contrast rank.",
        },
    }
    rows = rows[rows["run_id"].isin(run_metadata)].copy()
    if rows.empty:
        return pd.DataFrame()
    for run_id, metadata in run_metadata.items():
        mask = rows["run_id"].eq(run_id)
        for column, value in metadata.items():
            rows.loc[mask, column] = value

    rows["coverage"] = "all_available"
    rows["table_scope"] = "strict_qa_ensemble"
    rows["evidence_kind"] = "fixed_object_vote"
    rows["source_path"] = str(candidate_path)
    rows["selected_method"] = rows["run_id"].astype(str)
    rows["selection_rule"] = "fixed_object_vote"
    rows = rows.rename(
        columns={
            "mean_iou": "test_mean_iou",
            "mean_f1": "test_mean_f1",
        }
    )
    rows["validation_mean_iou"] = nan
    rows["validation_mean_f1"] = nan
    return rows[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _candidate_set_weight_rows(
    root: Path,
    *,
    experiment: str,
    ablation_id: str,
    method_label: str,
    phase2_role: str,
    evidence_kind: str,
    row_family: str,
    notes: str,
    table_scope: str = "strict_qa_ensemble",
) -> pd.DataFrame:
    comparison_root = root / "candidate_set_weight_control_comparison"
    selected_path = (
        comparison_root / f"{experiment}_validation_selected_dataset_summary.csv"
    )
    candidate_path = comparison_root / "candidate_set_weight_variant_fold_rows.csv"
    if not selected_path.exists() or not candidate_path.exists():
        return pd.DataFrame()

    selected = _read_csv(selected_path)
    candidate = _read_csv(candidate_path)
    candidate = candidate[
        candidate["experiment"].astype(str).eq(experiment)
        & candidate["split"].eq("test")
    ].copy()
    if candidate.empty:
        return pd.DataFrame()

    selected_runs = selected[
        ["dataset", "selected_method_variant", "validation_mean_iou"]
    ].drop_duplicates()
    rows = candidate.merge(
        selected_runs,
        left_on=["dataset", "method_variant"],
        right_on=["dataset", "selected_method_variant"],
        how="inner",
    )
    if rows.empty:
        return pd.DataFrame()

    rows["ablation_id"] = ablation_id
    rows["method_label"] = method_label
    rows["phase2_role"] = phase2_role
    rows["coverage"] = "all_available"
    rows["row_family"] = row_family
    rows["table_scope"] = table_scope
    rows["evidence_kind"] = evidence_kind
    rows["source_path"] = str(selected_path)
    rows["notes"] = notes
    rows["selected_method"] = rows["method_variant"].astype(str)
    rows["selection_rule"] = "validation_selected_method_variant_by_dataset"
    rows = rows.rename(
        columns={
            "mean_iou": "test_mean_iou",
            "mean_f1": "test_mean_f1",
        }
    )
    rows["validation_mean_f1"] = nan
    return rows[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _candidate_set_weight_fixed_comparison_rows(
    root: Path,
    *,
    rel_dir: str,
    ablation_id: str,
    method_label: str,
    phase2_role: str,
    evidence_kind: str,
    row_family: str,
    notes: str,
    selection_rule: str,
    table_scope: str = "strict_qa_ensemble",
) -> pd.DataFrame:
    path = root / rel_dir / "qa_weighted_object_vote_comparison_rows.csv"
    if not path.exists():
        return pd.DataFrame()

    rows = _read_csv(path)
    rows = rows[rows["selection_split"].astype(str).eq("test")].copy()
    if rows.empty:
        return pd.DataFrame()

    rows["ablation_id"] = ablation_id
    rows["method_label"] = method_label
    rows["phase2_role"] = phase2_role
    rows["coverage"] = "all_available"
    rows["row_family"] = row_family
    rows["table_scope"] = table_scope
    rows["evidence_kind"] = evidence_kind
    rows["source_path"] = str(path)
    rows["notes"] = notes
    rows["selected_method"] = rows["method_key"].astype(str)
    rows["selection_rule"] = selection_rule
    rows = rows.rename(
        columns={
            "mean_iou": "test_mean_iou",
            "mean_f1": "test_mean_f1",
        }
    )
    rows["validation_mean_iou"] = nan
    rows["validation_mean_f1"] = nan
    return rows[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _summarize_by_dataset(fold_rows: pd.DataFrame) -> pd.DataFrame:
    if fold_rows.empty:
        return pd.DataFrame()
    keys = [
        "ablation_id",
        "method_label",
        "phase2_role",
        "coverage",
        "row_family",
        "table_scope",
        "evidence_kind",
        "dataset",
    ]
    summary = (
        fold_rows.groupby(keys, dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_iou=("test_mean_iou", "mean"),
            mean_f1=("test_mean_f1", "mean"),
            mean_validation_iou=("validation_mean_iou", "mean"),
            selected_methods=("selected_method", lambda values: ";".join(sorted(set(map(str, values))))),
            notes=("notes", "first"),
        )
        .reset_index()
    )
    summary["missing_from_all_available"] = summary.apply(
        lambda row: ",".join(sorted(ALL_DATASETS - {row["dataset"]}))
        if row["coverage"] == "all_available" and row["dataset"] not in ALL_DATASETS
        else "",
        axis=1,
    )
    return summary


def _summarize_overall(dataset_rows: pd.DataFrame) -> pd.DataFrame:
    if dataset_rows.empty:
        return pd.DataFrame()
    keys = [
        "ablation_id",
        "method_label",
        "phase2_role",
        "coverage",
        "row_family",
        "table_scope",
        "evidence_kind",
    ]
    summary = (
        dataset_rows.groupby(keys, dropna=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            mean_validation_iou=("mean_validation_iou", "mean"),
            notes=("notes", "first"),
        )
        .reset_index()
    )
    summary["all_available_complete"] = summary["datasets"].eq(len(ALL_DATASETS)) & summary["coverage"].eq("all_available")
    return summary.sort_values(["coverage", "mean_iou"], ascending=[True, False])


def _context_dataset_rows(root: Path) -> pd.DataFrame:
    path = root / "ranker_pruned_fusion_probe" / "all_available_simple_policy_context_dataset_selected.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read_csv(path)
    df = df[df["fold"].eq("mean")].copy()
    df = df[~df["row_family"].eq("reporting_mode_selected")].copy()
    df["ablation_id"] = df["row_family"].map(
        {
            "full_pipeline_best_policy": "full_pipeline_best_policy",
            "fusion_only_best": "fusion_only_best",
            "competitor": "best_competitor",
            "silver_truth": "silver_truth",
        }
    ).fillna(df["method_key"].astype(str))
    df["phase2_role"] = df["row_family"].astype(str)
    df["coverage"] = "all_available"
    df["table_scope"] = "deployable_pipeline_context"
    df["evidence_kind"] = "phase1_context_dataset_mean"
    df["source_path"] = str(path)
    df["notes"] = df["row_family"].map(
        {
            "full_pipeline_best_policy": "QA-pruned fusion deployable pipeline context; FUSION remains a baseline component, not ENSEMBLE.",
            "reporting_mode_selected": "Validation-selected reporting context.",
            "fusion_only_best": "External fusion baseline context; never label as ours.",
            "competitor": "Best external competitor context.",
            "silver_truth": "Official silver truth target context.",
        }
    ).fillna("Context row.")
    df = df.rename(columns={"mean_iou": "test_mean_iou", "mean_f1": "test_mean_f1"})
    df["validation_mean_iou"] = nan
    df["validation_mean_f1"] = nan
    df["selected_method"] = df["source_method"]
    df["selection_rule"] = df["row_family"]
    return df[
        [
            "ablation_id",
            "method_label",
            "phase2_role",
            "coverage",
            "row_family",
            "table_scope",
            "evidence_kind",
            "dataset",
            "fold",
            "selected_method",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "source_path",
            "notes",
        ]
    ]


def _summarize_oracle_gaps(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("**/test_image_oracle_gap.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        parent = path.parent.name
        run_name = parent if parent != "balanced" else f"{path.parent.parent.parent.name}__{parent}"
        path_text = str(path)
        fold = "fold-1" if "fold1" in path_text else "fold-2" if "fold2" in path_text else ""
        dataset = "Fluo-C2DL-MSC" if "msc_" in path_text else ""
        rows.append(
            {
                "oracle_gap_id": run_name,
                "dataset": dataset,
                "fold": fold,
                "images": int(df["image_id"].nunique()) if "image_id" in df.columns else len(df),
                "mean_oracle_cluster_iou": df["oracle_mean_cluster_iou"].mean(),
                "mean_selected_cluster_iou": df["selected_mean_cluster_iou"].mean(),
                "mean_cluster_oracle_gap": df["mean_cluster_oracle_gap"].mean(),
                "max_cluster_oracle_gap": df["max_cluster_oracle_gap"].max(),
                "mean_image_max_candidate_iou": df["image_max_candidate_iou"].mean(),
                "mean_selected_max_candidate_iou": df["selected_max_candidate_iou"].mean(),
                "source_path": str(path),
                "notes": "MSC-only oracle diagnostic; cluster-level/image-level numbers are not final reconstructed test IoU.",
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(
    output_path: Path,
    validation_overall: pd.DataFrame,
    foldwise_best_overall: pd.DataFrame,
    coverage: pd.DataFrame,
    oracle_gaps: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 2 Soft-Hybrid Ablation Matrix",
        "",
        (
            "Generated from completed result artifacts. FUSION rows are "
            "context/baselines, not ENSEMBLE methods. Use `table_scope` to "
            "distinguish strict QA/ENSEMBLE evidence from diagnostic and "
            "context rows."
        ),
        "",
        "## Validation-Selected Performance",
        "",
    ]
    display_cols = [
        "method_label",
        "coverage",
        "row_family",
        "table_scope",
        "datasets",
        "mean_iou",
        "mean_f1",
    ]
    if not validation_overall.empty:
        lines.append(validation_overall[display_cols].to_markdown(index=False, floatfmt=".6f"))
    else:
        lines.append("_No validation-selected rows found._")
    lines.extend(["", "## Foldwise-Best Diagnostic", ""])
    if not foldwise_best_overall.empty:
        lines.append(foldwise_best_overall[display_cols].to_markdown(index=False, floatfmt=".6f"))
    else:
        lines.append("_No foldwise-best rows found._")
    lines.extend(["", "## Requested Phase 2 Coverage", ""])
    lines.append(coverage.to_markdown(index=False))
    lines.extend(["", "## Oracle Diagnostics", ""])
    if not oracle_gaps.empty:
        oracle_display = oracle_gaps[
            [
                "oracle_gap_id",
                "dataset",
                "fold",
                "mean_oracle_cluster_iou",
                "mean_selected_cluster_iou",
                "mean_cluster_oracle_gap",
            ]
        ]
        lines.append(oracle_display.to_markdown(index=False, floatfmt=".6f"))
    else:
        lines.append("_No oracle-gap diagnostics found._")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_tables(root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_parts = []
    foldwise_parts = []
    for run in STRATEGY_RUNS:
        run_dir = root / run.rel_dir
        if not run_dir.exists():
            continue
        selected_parts.append(_strategy_selected_rows(root, run))
        foldwise_parts.append(_strategy_foldwise_best_rows(root, run))

    selected_parts.append(_reporting_mode_selected_rows(root))
    foldwise_parts.append(_reporting_mode_foldwise_best_rows(root))
    selected_parts.append(_object_vote_fixed_rows(root))
    selected_parts.append(
        _object_vote_rows(
            root,
            selected_filename="object_vote_validation_selected_dataset_summary.csv",
            ablation_id="object_vote_validation_selected",
            method_label="Validation-selected internal object vote",
            phase2_role="internal_ensemble_validation_selected",
            evidence_kind="validation_selected_object_vote",
            selection_rule="validation_selected_by_dataset",
            notes=(
                "Strict internal ENSEMBLE row selecting uniform/boundary object-vote "
                "variants by validation IoU; no external fusion backend."
            ),
        )
    )
    selected_parts.append(
        _object_vote_rows(
            root,
            selected_filename="object_vote_modality_rule_dataset_summary.csv",
            ablation_id="object_vote_modality_rule",
            method_label="Modality-rule internal object vote",
            phase2_role="internal_ensemble_modality_rule",
            evidence_kind="modality_rule_object_vote",
            selection_rule="modality_rule",
            notes=(
                "Strict internal ENSEMBLE row using a simple deployable modality "
                "rule: BF uniform, DIC mild boundary, Fluo aggressive boundary."
            ),
        )
    )
    selected_parts.append(
        _candidate_set_weight_fixed_comparison_rows(
            root,
            rel_dir=(
                "candidate_set_weight_object_vote_mlp_trainonly_leave_target_video_out_thr050"
            ),
            ablation_id="candidate_set_weight_mlp_leave_target_video_out_thr050",
            method_label=(
                "Candidate-set MLP-weighted object vote, leave-target-video-out"
            ),
            phase2_role="candidate_set_learned_ensemble_leave_target_video_out",
            evidence_kind="candidate_set_weight_leave_target_video_out_fixed_thr050",
            row_family="ensemble_qa",
            selection_rule="fixed_threshold_0.50",
            notes=(
                "Strict internal ENSEMBLE row using per-fold candidate-set MLP "
                "models that exclude all rows from the held-out target video; "
                "fixed threshold 0.50."
            ),
        )
    )
    selected_parts.append(
        _candidate_set_weight_rows(
            root,
            experiment="mlp_trainonly_all_available",
            ablation_id="candidate_set_weight_mlp_trainonly",
            method_label="Candidate-set MLP-weighted object vote, train-only",
            phase2_role="candidate_set_learned_ensemble_trainonly",
            evidence_kind="candidate_set_weight_validation_selected",
            row_family="ensemble_qa",
            notes=(
                "Diagnostic all-available candidate-set MLP row using one "
                "global model trained on train split only. It can include "
                "reciprocal target-video train rows, so do not report as strict "
                "leave-target-video evidence."
            ),
            table_scope="diagnostic_qa_ensemble",
        )
    )
    selected_parts.append(
        _candidate_set_weight_rows(
            root,
            experiment="mlp_trainval_optimistic",
            ablation_id="candidate_set_weight_mlp_trainval_retrained",
            method_label=(
                "Candidate-set MLP-weighted object vote, train+validation refit"
            ),
            phase2_role="candidate_set_learned_ensemble_retrained",
            evidence_kind="candidate_set_weight_retrained_diagnostic",
            row_family="ensemble_qa_retrained",
            notes=(
                "Diagnostic all-available candidate-set MLP row using one "
                "global model trained on train+validation labels. It can include "
                "reciprocal target-video rows; use as retrained/headroom "
                "evidence, not as strict leave-target-video evidence."
            ),
            table_scope="diagnostic_qa_ensemble",
        )
    )
    selected_parts.append(
        _candidate_set_weight_rows(
            root,
            experiment="ridge_trainonly_all_available",
            ablation_id="candidate_set_weight_ridge_trainonly",
            method_label="Candidate-set ridge-weighted object vote, train-only",
            phase2_role="candidate_set_linear_control",
            evidence_kind="candidate_set_weight_linear_control",
            row_family="ensemble_qa",
            notes=(
                "Diagnostic all-available ridge control using one global "
                "candidate-set weight model trained on train split only. "
                "It can include reciprocal target-video rows, so do not report "
                "as strict leave-target-video evidence."
            ),
            table_scope="diagnostic_qa_ensemble",
        )
    )
    selected_parts.append(
        _candidate_set_weight_rows(
            root,
            experiment="mlp_trainonly_lodo",
            ablation_id="candidate_set_weight_mlp_lodo",
            method_label="Candidate-set MLP-weighted object vote, LODO",
            phase2_role="candidate_set_cross_dataset_generalization",
            evidence_kind="candidate_set_weight_lodo_stress_test",
            row_family="ensemble_qa",
            notes=(
                "Strict internal ENSEMBLE leave-one-dataset-out stress test; "
                "each dataset is scored by an MLP trained on the other datasets."
            ),
        )
    )
    selected_parts.append(
        _object_vote_fixed_run_rows(
            root,
            selection_rel_dir="candidate_set_weight_object_vote_selection_lodo_no_dataset",
            run_id="set_thr055",
            ablation_id="candidate_set_weight_mlp_lodo_no_dataset_fixed_thr055",
            method_label=(
                "Candidate-set MLP-weighted object vote, LODO no-dataset fixed threshold"
            ),
            phase2_role="candidate_set_cross_dataset_no_dataset_fixed_threshold",
            evidence_kind="candidate_set_weight_lodo_no_dataset_fixed_threshold",
            row_family="ensemble_qa",
            notes=(
                "Strict internal ENSEMBLE no-dataset LODO stress-test row using "
                "fixed set_thr055 after honest margin selection failed to beat "
                "the fixed threshold."
            ),
        )
    )
    selected_parts.append(
        _object_vote_rows(
            root,
            selection_rel_dir="candidate_set_weight_object_vote_selection_lodo_no_dataset",
            selected_filename=(
                "object_vote_leave_one_dataset_boundary_margin_selected_dataset_summary.csv"
            ),
            ablation_id="candidate_set_weight_mlp_lodo_no_dataset_boundary_margin",
            method_label=(
                "Candidate-set MLP-weighted object vote, LODO no-dataset boundary-margin"
            ),
            phase2_role="candidate_set_cross_dataset_no_dataset_boundary_margin",
            evidence_kind=(
                "candidate_set_weight_lodo_no_dataset_boundary_margin_validation_selected"
            ),
            selection_rule="leave_one_dataset_boundary_margin_selected",
            notes=(
                "Strict internal ENSEMBLE diagnostic: boundary-validation margin "
                "is selected using validation rows from other datasets, then "
                "applied to the held-out dataset. This non-test-selected protocol "
                "underperforms fixed set_thr055."
            ),
        )
    )

    selected_parts = [part for part in selected_parts if not part.empty]
    foldwise_parts = [part for part in foldwise_parts if not part.empty]
    selected_fold_rows = pd.concat(selected_parts, ignore_index=True)
    foldwise_best_fold_rows = pd.concat(foldwise_parts, ignore_index=True)

    selected_dataset = _summarize_by_dataset(selected_fold_rows)
    selected_context = _context_dataset_rows(root)
    if not selected_context.empty:
        selected_context_dataset = _summarize_by_dataset(selected_context)
        selected_dataset = pd.concat(
            [part for part in [selected_dataset, selected_context_dataset] if not part.empty],
            ignore_index=True,
        )

    selected_overall = _summarize_overall(selected_dataset)
    strict_dataset = selected_dataset[
        selected_dataset["table_scope"].eq("strict_qa_ensemble")
    ].copy()
    strict_overall = _summarize_overall(strict_dataset)

    foldwise_dataset = _summarize_by_dataset(foldwise_best_fold_rows)
    foldwise_overall = _summarize_overall(foldwise_dataset)

    coverage = pd.DataFrame(REQUESTED_ABLATIONS)
    oracle_gaps = _summarize_oracle_gaps(root)

    selected_fold_rows.to_csv(output_dir / "phase2_validation_selected_fold_rows.csv", index=False)
    selected_dataset.to_csv(output_dir / "phase2_validation_selected_dataset_summary.csv", index=False)
    selected_overall.to_csv(output_dir / "phase2_validation_selected_overall_summary.csv", index=False)
    strict_dataset.to_csv(output_dir / "phase2_strict_qa_ensemble_dataset_summary.csv", index=False)
    strict_overall.to_csv(output_dir / "phase2_strict_qa_ensemble_overall_summary.csv", index=False)
    foldwise_best_fold_rows.to_csv(output_dir / "phase2_foldwise_best_fold_rows.csv", index=False)
    foldwise_dataset.to_csv(output_dir / "phase2_foldwise_best_dataset_summary.csv", index=False)
    foldwise_overall.to_csv(output_dir / "phase2_foldwise_best_overall_summary.csv", index=False)
    coverage.to_csv(output_dir / "phase2_requested_ablation_coverage.csv", index=False)
    oracle_gaps.to_csv(output_dir / "phase2_oracle_gap_msc_summary.csv", index=False)
    _write_markdown(
        output_dir / "phase2_ablation_matrix.md",
        selected_overall,
        foldwise_overall,
        coverage,
        oracle_gaps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("hpc_results/soft_hybrid_probe"),
        help="Root directory containing soft-hybrid result artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpc_results/soft_hybrid_probe/phase2_ablation_matrix"),
        help="Directory for Phase 2 ablation tables.",
    )
    args = parser.parse_args()
    build_tables(args.root, args.output_dir)


if __name__ == "__main__":
    main()
