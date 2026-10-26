from pathlib import Path

import pandas as pd

from scripts.build_candidate_set_weight_selection_calibration_tables import (
    build_dataset_table,
    build_overall_table,
    build_tables,
)


def test_candidate_set_selection_calibration_report_separates_context(
    tmp_path: Path,
) -> None:
    root = tmp_path
    phase2 = root / "phase2_ablation_matrix"
    phase2.mkdir()
    pd.DataFrame(
        [
            {
                "ablation_id": "candidate_set_weight_mlp_trainonly",
                "method_label": "Candidate-set MLP",
                "row_family": "ensemble_qa",
                "evidence_kind": "candidate_set_weight_validation_selected",
                "datasets": 1,
                "mean_iou": 0.87,
                "mean_f1": 0.93,
                "mean_validation_iou": 0.86,
                "notes": "strict",
            }
        ]
    ).to_csv(phase2 / "phase2_strict_qa_ensemble_overall_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "ablation_id": "candidate_set_weight_mlp_trainonly",
                "method_label": "Candidate-set MLP",
                "row_family": "ensemble_qa",
                "evidence_kind": "candidate_set_weight_validation_selected",
                "dataset": "Toy",
                "mean_iou": 0.87,
                "mean_f1": 0.93,
                "mean_validation_iou": 0.86,
                "selected_methods": "set_thr050",
                "notes": "strict",
            }
        ]
    ).to_csv(phase2 / "phase2_strict_qa_ensemble_dataset_summary.csv", index=False)

    selector = root / "candidate_set_weight_object_vote_selection_trainonly_all_available"
    selector.mkdir()
    pd.DataFrame(
        [
            {
                "selection_rule": "validation_selected_by_dataset",
                "datasets": 1,
                "mean_iou": 0.87,
                "mean_f1": 0.93,
                "selected_runs": "Toy=set_thr050",
            },
            {
                "selection_rule": "fixed::set_thr055",
                "datasets": 1,
                "mean_iou": 0.86,
                "mean_f1": 0.92,
                "selected_runs": "Toy=set_thr055",
            },
        ]
    ).to_csv(selector / "object_vote_selection_overall_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "Toy",
                "run_id": "set_thr050",
                "mean_iou": 0.87,
                "mean_f1": 0.93,
                "validation_mean_iou": 0.86,
            }
        ]
    ).to_csv(
        selector / "object_vote_validation_selected_dataset_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "dataset": "Toy",
                "run_id": "set_thr055",
                "split": "test",
                "mean_iou": 0.86,
                "mean_f1": 0.92,
            }
        ]
    ).to_csv(selector / "object_vote_candidate_dataset_summary.csv", index=False)

    evolution = root / "qa_ensemble_evolution"
    evolution.mkdir()
    pd.DataFrame(
        [
            {
                "comparison_group": "deployable_context",
                "method_id": "full_pipeline_best_policy",
                "method_label": "Deployable",
                "row_family": "full_pipeline_best_policy",
                "table_scope": "context_not_strict_qa_ensemble",
                "datasets": 1,
                "mean_iou": 0.88,
                "mean_f1": 0.94,
                "notes": "context",
            },
            {
                "comparison_group": "external_context",
                "method_id": "fusion_only_best",
                "method_label": "FUSION baseline",
                "row_family": "fusion_only_best",
                "table_scope": "context_not_strict_qa_ensemble",
                "datasets": 1,
                "mean_iou": 0.80,
                "notes": "baseline",
            },
        ]
    ).to_csv(evolution / "qa_ensemble_evolution_overall_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "comparison_group": "deployable_context",
                "method_id": "full_pipeline_best_policy",
                "method_label": "Deployable",
                "row_family": "full_pipeline_best_policy",
                "table_scope": "context_not_strict_qa_ensemble",
                "dataset": "Toy",
                "mean_iou": 0.88,
                "mean_f1": 0.94,
                "selected_methods": "full_pipeline",
                "notes": "context",
            },
            {
                "comparison_group": "external_context",
                "method_id": "fusion_only_best",
                "method_label": "FUSION baseline",
                "row_family": "fusion_only_best",
                "table_scope": "context_not_strict_qa_ensemble",
                "dataset": "Toy",
                "mean_iou": 0.80,
                "selected_methods": "fusion",
                "notes": "baseline",
            },
        ]
    ).to_csv(evolution / "qa_ensemble_evolution_dataset_summary.csv", index=False)

    overall = build_overall_table(root)
    dataset = build_dataset_table(root)

    assert "strict_qa_ensemble" in set(overall["comparison_group"])
    assert "selector_diagnostic" in set(overall["comparison_group"])
    assert "deployable_context" in set(overall["comparison_group"])
    assert "fusion_baseline_context" in set(overall["comparison_group"])
    assert (
        overall.set_index("method_id").loc["fusion_only_best", "row_family"]
        == "fusion_only_best"
    )
    assert set(dataset["dataset"]) == {"Toy"}

    build_tables(root=root, output_dir=root / "out")
    assert (
        root / "out" / "candidate_set_weight_selection_calibration_summary.md"
    ).exists()
