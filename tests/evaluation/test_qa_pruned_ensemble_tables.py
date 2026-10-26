import pandas as pd
import pytest

from scripts.qa_pruned_ensemble.common import (
    filter_strict_ensemble_rows,
    summarize_overall,
)
from scripts.qa_pruned_ensemble.build_validation_selection_tables import (
    _run_id,
    select_by_leave_one_dataset_validation_mean_minus_std,
    select_by_validation_maximin,
    select_by_validation_mean_minus_std,
    select_by_guarded_margin,
    select_by_global_validation,
    select_by_leave_one_dataset_validation,
    select_by_validation,
    summarize_selected_outputs,
)
from scripts.qa_pruned_ensemble.run_pruned_candidate_set_weight import (
    _calibrated_weight_values,
    discover_prepared_candidate_jobs,
)
from scripts.qa_pruned_ensemble.build_final_paper_tables import (
    collect_selected_fold_rows,
    summarize_pairwise_deltas,
    summarize_selected_fold_rows,
)


def test_summarize_overall_includes_dataset_policy_scope() -> None:
    dataset_summary = pd.DataFrame(
        [
            {
                "result_family": "qa_pruned_ensemble_hard_gate",
                "row_family": "ensemble_qa",
                "method_key": "ensemble_qa_pruned_hard_gate_weighted_object_vote",
                "method_label": "QA-pruned QA-weighted ENSEMBLE",
                "ensemble_variant": "qa_weighted",
                "qa_pruning_policy": "source_balanced_topk",
                "dataset": "Toy-A",
                "mean_iou": 0.9,
                "mean_f1": 0.95,
                "n_folds": 2,
                "n_cells": 10,
                "mean_candidates_used": 3.0,
            },
            {
                "result_family": "qa_pruned_ensemble_hard_gate",
                "row_family": "ensemble_qa",
                "method_key": "ensemble_qa_pruned_hard_gate_weighted_object_vote",
                "method_label": "QA-pruned QA-weighted ENSEMBLE",
                "ensemble_variant": "qa_weighted",
                "qa_pruning_policy": "top1_per_cluster",
                "dataset": "Toy-B",
                "mean_iou": 0.7,
                "mean_f1": 0.8,
                "n_folds": 2,
                "n_cells": 8,
                "mean_candidates_used": 1.0,
            },
        ]
    )

    overall = summarize_overall(dataset_summary)
    dataset_policy = overall[
        overall["selection_scope"].eq("dataset_policy")
        & overall["method_key"].eq("ensemble_qa_pruned_hard_gate_weighted_object_vote")
    ].iloc[0]

    assert dataset_policy["qa_pruning_policy"] == "dataset_policy"
    assert float(dataset_policy["mean_iou"]) == 0.8
    assert int(dataset_policy["n_datasets"]) == 2
    assert int(dataset_policy["n_folds"]) == 4


def test_filter_strict_ensemble_rows_excludes_full_pipeline_context() -> None:
    rows = pd.DataFrame(
        [
            {"row_family": "ensemble_qa", "method_key": "ours"},
            {"row_family": "full_pipeline", "method_key": "qa_pruned_simple"},
            {"row_family": "fusion_only", "method_key": "simple_baseline"},
        ]
    )

    strict = filter_strict_ensemble_rows(rows)

    assert strict["method_key"].tolist() == ["ours"]


def test_validation_selection_uses_validation_split_for_run_choice() -> None:
    rows = pd.DataFrame(
        [
            {"dataset": "A", "run_id": "r1", "split": "validation", "mean_iou": 0.9, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "r2", "split": "validation", "mean_iou": 0.8, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "r1", "split": "test", "mean_iou": 0.7, "mean_f1": 0.8, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "r2", "split": "test", "mean_iou": 0.95, "mean_f1": 0.97, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "r1", "split": "validation", "mean_iou": 0.65, "mean_f1": 0.7, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "r2", "split": "validation", "mean_iou": 0.7, "mean_f1": 0.8, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "r1", "split": "test", "mean_iou": 0.5, "mean_f1": 0.6, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "r2", "split": "test", "mean_iou": 0.8, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
        ]
    )

    selected = select_by_validation(rows)
    global_selected = select_by_global_validation(rows)
    lodo_selected = select_by_leave_one_dataset_validation(rows)
    overall = summarize_selected_outputs(
        {
            "validation_selected_by_dataset": selected,
            "global_validation_selected": global_selected,
            "leave_one_dataset_validation_selected": lodo_selected,
        }
    )

    assert selected.set_index("dataset")["run_id"].to_dict() == {"A": "r1", "B": "r2"}
    assert global_selected["run_id"].unique().tolist() == ["r1"]
    assert lodo_selected.set_index("dataset")["run_id"].to_dict() == {"A": "r2", "B": "r1"}
    assert float(
        overall.set_index("selection_rule").loc[
            "validation_selected_by_dataset", "mean_iou"
        ]
    ) == 0.75


def test_guarded_margin_keeps_baseline_when_validation_gain_is_small() -> None:
    rows = pd.DataFrame(
        [
            {"dataset": "A", "run_id": "base", "split": "validation", "mean_iou": 0.90, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "switch", "split": "validation", "mean_iou": 0.905, "mean_f1": 0.91, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "base", "split": "test", "mean_iou": 0.80, "mean_f1": 0.85, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "switch", "split": "test", "mean_iou": 0.70, "mean_f1": 0.80, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "base", "split": "validation", "mean_iou": 0.80, "mean_f1": 0.8, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "switch", "split": "validation", "mean_iou": 0.83, "mean_f1": 0.84, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "base", "split": "test", "mean_iou": 0.60, "mean_f1": 0.70, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "switch", "split": "test", "mean_iou": 0.90, "mean_f1": 0.95, "n_folds": 1, "n_cells": 5},
        ]
    )

    selected = select_by_guarded_margin(
        rows,
        baseline_run_id="base",
        validation_margin=0.01,
    )

    assert selected.set_index("dataset")["run_id"].to_dict() == {
        "A": "base",
        "B": "switch",
    }
    assert float(selected["mean_iou"].mean()) == pytest.approx(0.85)


def test_validation_run_id_distinguishes_boundary_settings() -> None:
    base = {
        "result_family": "qa_pruned_ensemble_soft_gate",
        "method_key": "ensemble_qa_pruned_soft_gate_clipped_score_object_vote",
        "ensemble_variant": "clipped_score",
        "qa_pruning_policy": "source_balanced_topk",
        "soft_gate": "clipped_score",
        "weights_column": "_qa_soft_gate_weight",
        "boundary_support_column": "local_image_support_score",
        "boundary_floor_column": "",
        "boundary_iterations": 1,
        "threshold": 0.5,
    }

    floor_060 = _run_id(pd.Series({**base, "boundary_floor": 0.60}))
    floor_090 = _run_id(pd.Series({**base, "boundary_floor": 0.90}))

    assert floor_060 != floor_090


def test_pruned_candidate_set_weight_discovers_prepared_jobs(tmp_path) -> None:
    candidate_run = "clipped_score_boundary_floor075_thr050"
    prepared = (
        tmp_path
        / "BF-C2DL-HSC"
        / "sz64"
        / "baseline"
        / "fold-1"
        / "source_balanced_topk"
        / candidate_run
        / "prepared_soft_gate_candidates.parquet"
    )
    prepared.parent.mkdir(parents=True)
    prepared.touch()

    jobs = discover_prepared_candidate_jobs(tmp_path, candidate_run)

    assert jobs == [
        {
            "dataset": "BF-C2DL-HSC",
            "crop_tag": "sz64",
            "variant": "baseline",
            "fold": "fold-1",
            "policy": "source_balanced_topk",
            "candidate_run": candidate_run,
            "soft_gate": "clipped_score",
            "prepared_parquet": prepared,
        }
    ]


def test_pruned_candidate_set_weight_cluster_softmax_preserves_winner() -> None:
    frame = pd.DataFrame(
        {
            "ranker_cluster_id": ["a", "a", "b", "b"],
            "candidate_set_weight_raw_score": [0.0, 2.0, 4.0, 4.0],
        }
    )

    weights = _calibrated_weight_values(
        frame,
        weight_transform="raw_cluster_softmax",
        weight_temperature=1.0,
        min_score=1e-3,
    )

    assert weights.iloc[1] == 1.0
    assert weights.iloc[0] < weights.iloc[1]
    assert weights.iloc[2] == weights.iloc[3]


def test_validation_stability_selectors_require_full_coverage() -> None:
    rows = pd.DataFrame(
        [
            {"dataset": "A", "run_id": "high_mean", "split": "validation", "mean_iou": 0.95, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "high_mean", "split": "validation", "mean_iou": 0.70, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "stable", "split": "validation", "mean_iou": 0.86, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "stable", "split": "validation", "mean_iou": 0.84, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "partial", "split": "validation", "mean_iou": 0.99, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "high_mean", "split": "test", "mean_iou": 0.80, "mean_f1": 0.8, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "high_mean", "split": "test", "mean_iou": 0.80, "mean_f1": 0.8, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "stable", "split": "test", "mean_iou": 0.90, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "stable", "split": "test", "mean_iou": 0.90, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "partial", "split": "test", "mean_iou": 0.99, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
        ]
    )

    mean_minus_std = select_by_validation_mean_minus_std(rows, std_penalty=1.0)
    maximin = select_by_validation_maximin(rows)

    assert mean_minus_std["run_id"].unique().tolist() == ["stable"]
    assert maximin["run_id"].unique().tolist() == ["stable"]


def test_leave_one_dataset_stability_selector_scores_other_datasets() -> None:
    rows = pd.DataFrame(
        [
            {"dataset": "A", "run_id": "r1", "split": "validation", "mean_iou": 0.90, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "r1", "split": "validation", "mean_iou": 0.70, "mean_f1": 0.8, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "r2", "split": "validation", "mean_iou": 0.80, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "r2", "split": "validation", "mean_iou": 0.85, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "r1", "split": "test", "mean_iou": 0.60, "mean_f1": 0.7, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "r1", "split": "test", "mean_iou": 0.70, "mean_f1": 0.8, "n_folds": 1, "n_cells": 5},
            {"dataset": "A", "run_id": "r2", "split": "test", "mean_iou": 0.80, "mean_f1": 0.9, "n_folds": 1, "n_cells": 5},
            {"dataset": "B", "run_id": "r2", "split": "test", "mean_iou": 0.90, "mean_f1": 0.95, "n_folds": 1, "n_cells": 5},
        ]
    )

    selected = select_by_leave_one_dataset_validation_mean_minus_std(
        rows,
        std_penalty=1.0,
    )

    assert selected.set_index("dataset")["run_id"].to_dict() == {
        "A": "r2",
        "B": "r1",
    }


def test_final_paper_tables_join_selected_runs_to_test_folds() -> None:
    fold_rows = pd.DataFrame(
        [
            {"dataset": "A", "fold": "fold-1", "run_id": "base", "split": "test", "mean_iou": 0.80, "mean_f1": 0.90, "n_cells": 10},
            {"dataset": "A", "fold": "fold-2", "run_id": "base", "split": "test", "mean_iou": 0.82, "mean_f1": 0.91, "n_cells": 10},
            {"dataset": "A", "fold": "fold-1", "run_id": "candidate", "split": "test", "mean_iou": 0.86, "mean_f1": 0.93, "n_cells": 10},
            {"dataset": "A", "fold": "fold-2", "run_id": "candidate", "split": "test", "mean_iou": 0.88, "mean_f1": 0.94, "n_cells": 10},
            {"dataset": "A", "fold": "fold-1", "run_id": "candidate", "split": "validation", "mean_iou": 0.99, "mean_f1": 0.99, "n_cells": 3},
            {"dataset": "B", "fold": "fold-1", "run_id": "base", "split": "test", "mean_iou": 0.70, "mean_f1": 0.80, "n_cells": 10},
            {"dataset": "B", "fold": "fold-2", "run_id": "base", "split": "test", "mean_iou": 0.72, "mean_f1": 0.81, "n_cells": 10},
            {"dataset": "B", "fold": "fold-1", "run_id": "candidate", "split": "test", "mean_iou": 0.74, "mean_f1": 0.82, "n_cells": 10},
            {"dataset": "B", "fold": "fold-2", "run_id": "candidate", "split": "test", "mean_iou": 0.76, "mean_f1": 0.83, "n_cells": 10},
        ]
    )
    selected_dataset_rows = pd.DataFrame(
        [
            {"selection_rule": "global_validation_selected", "dataset": "A", "run_id": "base", "method_label": "Base ENSEMBLE", "ensemble_variant": "base", "soft_gate": "score", "threshold": 0.5},
            {"selection_rule": "global_validation_selected", "dataset": "B", "run_id": "base", "method_label": "Base ENSEMBLE", "ensemble_variant": "base", "soft_gate": "score", "threshold": 0.5},
            {"selection_rule": "stability_selected", "dataset": "A", "run_id": "candidate", "method_label": "Candidate ENSEMBLE", "ensemble_variant": "candidate", "soft_gate": "score", "threshold": 0.48},
            {"selection_rule": "stability_selected", "dataset": "B", "run_id": "candidate", "method_label": "Candidate ENSEMBLE", "ensemble_variant": "candidate", "soft_gate": "score", "threshold": 0.48},
        ]
    )

    selected_folds = collect_selected_fold_rows(
        fold_rows=fold_rows,
        selected_dataset_rows=selected_dataset_rows,
    )
    overall = summarize_selected_fold_rows(
        selected_folds,
        selected_dataset_rows=selected_dataset_rows,
        bootstrap_samples=0,
        bootstrap_seed=1,
    )
    deltas = summarize_pairwise_deltas(
        selected_folds,
        reference_rule="global_validation_selected",
        bootstrap_samples=0,
        bootstrap_seed=1,
    )

    assert selected_folds["split"].unique().tolist() == ["test"]
    assert len(selected_folds) == 8
    assert float(
        overall.set_index("selection_rule").loc["stability_selected", "mean_iou"]
    ) == pytest.approx(0.81)
    assert float(
        deltas.set_index("candidate_rule").loc["stability_selected", "delta_iou"]
    ) == pytest.approx(0.05)
