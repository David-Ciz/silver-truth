from pathlib import Path

import pandas as pd

from scripts.build_qa_ensemble_evolution_tables import (
    build_dataset_table,
    build_overall_table,
    build_tables,
)


def test_qa_ensemble_evolution_table_separates_strict_legacy_and_context(
    tmp_path: Path,
) -> None:
    phase2_dir = tmp_path / "phase2"
    phase2_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ablation_id": "candidate_set_weight_mlp_leave_target_video_out_thr050",
                "method_label": "Candidate-set MLP, leave-target-video-out",
                "phase2_role": "candidate_set_learned_ensemble_leave_target_video_out",
                "evidence_kind": "candidate_set_weight_leave_target_video_out_fixed_thr050",
                "row_family": "ensemble_qa",
                "table_scope": "strict_qa_ensemble",
                "datasets": 1,
                "mean_iou": 0.87,
                "mean_f1": 0.93,
                "mean_validation_iou": 0.86,
                "notes": "strict",
            },
            {
                "ablation_id": "candidate_set_weight_mlp_lodo",
                "method_label": "Candidate-set MLP, LODO",
                "phase2_role": "candidate_set_cross_dataset_generalization",
                "evidence_kind": "candidate_set_weight_lodo_stress_test",
                "row_family": "ensemble_qa",
                "table_scope": "strict_qa_ensemble",
                "datasets": 1,
                "mean_iou": 0.88,
                "mean_f1": 0.94,
                "mean_validation_iou": 0.87,
                "notes": "Strict internal ENSEMBLE leave-one-dataset-out stress test.",
            }
        ]
    ).to_csv(phase2_dir / "phase2_strict_qa_ensemble_overall_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "ablation_id": "candidate_set_weight_mlp_leave_target_video_out_thr050",
                "method_label": "Candidate-set MLP, leave-target-video-out",
                "phase2_role": "candidate_set_learned_ensemble_leave_target_video_out",
                "evidence_kind": "candidate_set_weight_leave_target_video_out_fixed_thr050",
                "row_family": "ensemble_qa",
                "table_scope": "strict_qa_ensemble",
                "dataset": "Toy",
                "mean_iou": 0.87,
                "mean_f1": 0.93,
                "mean_validation_iou": 0.86,
                "selected_methods": "set_thr050",
                "notes": "strict",
            },
            {
                "ablation_id": "candidate_set_weight_mlp_lodo",
                "method_label": "Candidate-set MLP, LODO",
                "phase2_role": "candidate_set_cross_dataset_generalization",
                "evidence_kind": "candidate_set_weight_lodo_stress_test",
                "row_family": "ensemble_qa",
                "table_scope": "strict_qa_ensemble",
                "dataset": "Toy",
                "mean_iou": 0.88,
                "mean_f1": 0.94,
                "mean_validation_iou": 0.87,
                "selected_methods": "set_thr055",
                "notes": "Strict internal ENSEMBLE leave-one-dataset-out stress test.",
            }
        ]
    ).to_csv(phase2_dir / "phase2_strict_qa_ensemble_dataset_summary.csv", index=False)

    legacy = tmp_path / "legacy.csv"
    pd.DataFrame(
        [
            {
                "dataset": "Toy",
                "row_type": "mean",
                "validation_selected_ours_iou": 0.70,
                "validation_selected_validation_iou": 0.71,
                "validation_selected_ours_method": "ensemble_qa_t0.50",
                "best_ours_iou": 0.72,
                "best_ours_method": "foldwise best qa/ensemble-only",
            }
        ]
    ).to_csv(legacy, index=False)

    context_overall = tmp_path / "context_overall.csv"
    pd.DataFrame(
        [
            {
                "row_family": "full_pipeline_best_policy",
                "datasets": 1,
                "mean_iou": 0.86,
                "mean_f1": 0.92,
            },
            {
                "row_family": "fusion_only_best",
                "datasets": 1,
                "mean_iou": 0.80,
            },
        ]
    ).to_csv(context_overall, index=False)

    context_dataset = tmp_path / "context_dataset.csv"
    pd.DataFrame(
        [
            {
                "dataset": "Toy",
                "fold": "mean",
                "row_family": "full_pipeline_best_policy",
                "method_key": "full_pipeline__qa_pruned_simple",
                "mean_iou": 0.86,
                "mean_f1": 0.92,
            },
            {
                "dataset": "Toy",
                "fold": "mean",
                "row_family": "fusion_only_best",
                "method_key": "fusion_only__simple",
                "mean_iou": 0.80,
            },
        ]
    ).to_csv(context_dataset, index=False)

    overall = build_overall_table(
        phase2_dir=phase2_dir,
        legacy_summary=legacy,
        context_summary=context_overall,
    )
    dataset = build_dataset_table(
        phase2_dir=phase2_dir,
        legacy_summary=legacy,
        context_dataset=context_dataset,
    )

    groups = set(overall["comparison_group"])
    strict = overall.set_index("method_id").loc[
        "candidate_set_weight_mlp_leave_target_video_out_thr050"
    ]
    stress = overall.set_index("method_id").loc["candidate_set_weight_mlp_lodo"]
    context = overall.set_index("method_id").loc["fusion_only_best"]

    assert groups == {
        "strict_fold_protocol",
        "strict_cross_dataset_stress",
        "legacy_simple_nn",
        "deployable_context",
        "external_context",
    }
    assert strict["comparison_group"] == "strict_fold_protocol"
    assert strict["table_scope"] == "strict_qa_ensemble"
    assert stress["comparison_group"] == "strict_cross_dataset_stress"
    assert context["table_scope"] == "context_not_strict_qa_ensemble"
    assert set(dataset["dataset"]) == {"Toy"}

    build_tables(
        phase2_dir=phase2_dir,
        legacy_summary=legacy,
        context_summary=context_overall,
        context_dataset=context_dataset,
        output_dir=tmp_path / "out",
    )
    assert (tmp_path / "out" / "qa_ensemble_evolution_summary.md").exists()


def test_qa_ensemble_evolution_includes_badmass_selection_rows(
    tmp_path: Path,
) -> None:
    phase2_dir = tmp_path / "phase2"
    phase2_dir.mkdir()
    pd.DataFrame(
        columns=[
            "ablation_id",
            "method_label",
            "phase2_role",
            "evidence_kind",
            "row_family",
            "table_scope",
            "datasets",
            "mean_iou",
            "mean_f1",
            "mean_validation_iou",
            "notes",
        ]
    ).to_csv(phase2_dir / "phase2_strict_qa_ensemble_overall_summary.csv", index=False)
    pd.DataFrame(
        columns=[
            "ablation_id",
            "method_label",
            "phase2_role",
            "evidence_kind",
            "row_family",
            "table_scope",
            "dataset",
            "mean_iou",
            "mean_f1",
            "mean_validation_iou",
            "selected_methods",
            "notes",
        ]
    ).to_csv(phase2_dir / "phase2_strict_qa_ensemble_dataset_summary.csv", index=False)

    badmass_dir = tmp_path / "badmass"
    badmass_dir.mkdir()
    pd.DataFrame(
        [
            {
                "selection_rule": "validation_selected_by_dataset",
                "mean_iou": 0.8615,
                "mean_f1": 0.9200,
                "datasets": 1,
                "selected_runs": "Toy=deep_dense_thr055",
            },
            {
                "selection_rule": "fixed::badmass_w010_topk5_thr055",
                "mean_iou": 0.8609,
                "mean_f1": 0.9194,
                "datasets": 1,
                "selected_runs": "Toy=badmass_w010_topk5_thr055",
            },
            {
                "selection_rule": "fixed::mlp_thr050",
                "mean_iou": 0.8607,
                "mean_f1": 0.9207,
                "datasets": 1,
                "selected_runs": "Toy=mlp_thr050",
            },
        ]
    ).to_csv(badmass_dir / "object_vote_selection_overall_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "deep_dense_thr055",
                "dataset": "Toy",
                "mean_iou": 0.8615,
                "mean_f1": 0.9200,
                "validation_mean_iou": 0.8700,
            }
        ]
    ).to_csv(
        badmass_dir / "object_vote_validation_selected_dataset_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "run_id": "badmass_w010_topk5_thr055",
                "dataset": "Toy",
                "split": "overall",
                "mean_iou": 0.9000,
                "mean_f1": 0.9500,
            },
            {
                "run_id": "badmass_w010_topk5_thr055",
                "dataset": "Toy",
                "split": "test",
                "mean_iou": 0.8609,
                "mean_f1": 0.9194,
            },
            {
                "run_id": "badmass_w010_topk5_thr055",
                "dataset": "Toy",
                "split": "validation",
                "mean_iou": 0.8700,
                "mean_f1": 0.9300,
            },
        ]
    ).to_csv(badmass_dir / "object_vote_candidate_dataset_summary.csv", index=False)

    legacy = tmp_path / "legacy.csv"
    pd.DataFrame(
        columns=[
            "dataset",
            "row_type",
            "validation_selected_ours_iou",
            "validation_selected_validation_iou",
            "validation_selected_ours_method",
            "best_ours_iou",
            "best_ours_method",
        ]
    ).to_csv(legacy, index=False)

    context_overall = tmp_path / "context_overall.csv"
    pd.DataFrame(columns=["row_family", "datasets", "mean_iou", "mean_f1"]).to_csv(
        context_overall, index=False
    )
    context_dataset = tmp_path / "context_dataset.csv"
    pd.DataFrame(
        columns=["dataset", "fold", "row_family", "method_key", "mean_iou", "mean_f1"]
    ).to_csv(context_dataset, index=False)

    overall = build_overall_table(
        phase2_dir=phase2_dir,
        legacy_summary=legacy,
        context_summary=context_overall,
        badmass_selection_dir=badmass_dir,
    )
    dataset = build_dataset_table(
        phase2_dir=phase2_dir,
        legacy_summary=legacy,
        context_dataset=context_dataset,
        badmass_selection_dir=badmass_dir,
    )

    overall_by_id = overall.set_index("method_id")
    selected_id = "candidate_set_weight_deep_sets_badmass_validation_selected_by_dataset"
    fixed_id = "candidate_set_weight_deep_sets_badmass_w010_topk5_thr055_fixed"

    assert selected_id in overall_by_id.index
    assert fixed_id in overall_by_id.index
    assert "fixed::mlp_thr050" not in set(overall["method_id"])
    assert overall_by_id.loc[selected_id, "row_family"] == "ensemble_qa"
    assert overall_by_id.loc[selected_id, "table_scope"] == "strict_qa_ensemble"

    fixed_dataset = dataset.set_index("method_id").loc[fixed_id]
    assert fixed_dataset["mean_iou"] == 0.8609
    assert fixed_dataset["mean_validation_iou"] == 0.8700
