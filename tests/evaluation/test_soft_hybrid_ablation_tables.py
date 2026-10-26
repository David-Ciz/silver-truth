from pathlib import Path

import pandas as pd

from scripts.build_soft_hybrid_ablation_tables import (
    _candidate_set_weight_fixed_comparison_rows,
    _candidate_set_weight_rows,
    _object_vote_fixed_run_rows,
    _object_vote_fixed_rows,
    _object_vote_rows,
    _summarize_by_dataset,
    _summarize_overall,
    _write_markdown,
)


def test_object_vote_rows_are_strict_qa_ensemble_taxonomy(tmp_path: Path) -> None:
    root = tmp_path
    table_dir = root / "object_vote_selection_tables"
    table_dir.mkdir()
    pd.DataFrame(
        [
            {
                "run_id": "uniform_thr050",
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.8,
                "mean_f1": 0.9,
            },
            {
                "run_id": "boundary_full_floor025",
                "dataset": "Fluo-C2DL-MSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.7,
                "mean_f1": 0.85,
            },
        ]
    ).to_csv(table_dir / "object_vote_candidate_fold_rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "uniform_thr050",
                "dataset": "BF-C2DL-HSC",
                "mean_iou": 0.8,
                "mean_f1": 0.9,
                "validation_mean_iou": 0.82,
            },
            {
                "run_id": "boundary_full_floor025",
                "dataset": "Fluo-C2DL-MSC",
                "mean_iou": 0.7,
                "mean_f1": 0.85,
                "validation_mean_iou": 0.72,
            },
        ]
    ).to_csv(
        table_dir / "object_vote_validation_selected_dataset_summary.csv",
        index=False,
    )

    selected = _object_vote_rows(
        root,
        selected_filename="object_vote_validation_selected_dataset_summary.csv",
        ablation_id="object_vote_validation_selected",
        method_label="Validation-selected internal object vote",
        phase2_role="internal_ensemble_validation_selected",
        evidence_kind="validation_selected_object_vote",
        selection_rule="validation_selected_by_dataset",
        notes="test",
    )

    assert set(selected["row_family"]) == {"ensemble_qa"}
    assert set(selected["table_scope"]) == {"strict_qa_ensemble"}
    assert selected["test_mean_iou"].tolist() == [0.8, 0.7]
    assert selected["validation_mean_iou"].tolist() == [0.82, 0.72]


def test_object_vote_fixed_rows_include_uniform_as_ensemble_only(tmp_path: Path) -> None:
    root = tmp_path
    table_dir = root / "object_vote_selection_tables"
    table_dir.mkdir()
    pd.DataFrame(
        [
            {
                "run_id": "uniform_thr050",
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.8,
                "mean_f1": 0.9,
            },
            {
                "run_id": "boundary_full_floor075",
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.81,
                "mean_f1": 0.91,
            },
        ]
    ).to_csv(table_dir / "object_vote_candidate_fold_rows.csv", index=False)

    fixed = _object_vote_fixed_rows(root)
    dataset = _summarize_by_dataset(fixed)
    overall = _summarize_overall(dataset)
    families = fixed.set_index("ablation_id")["row_family"].to_dict()

    assert families["object_vote_uniform"] == "ensemble_only"
    assert families["object_vote_boundary_floor075"] == "ensemble_qa"
    assert set(fixed["table_scope"]) == {"strict_qa_ensemble"}
    assert overall.set_index("ablation_id").loc["object_vote_uniform", "mean_iou"] == 0.8


def test_candidate_set_weight_rows_are_strict_qa_ensemble_taxonomy(
    tmp_path: Path,
) -> None:
    root = tmp_path
    table_dir = root / "candidate_set_weight_control_comparison"
    table_dir.mkdir()
    pd.DataFrame(
        [
            {
                "experiment": "mlp_trainonly_all_available",
                "method_variant": "set_thr050",
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.82,
                "mean_f1": 0.91,
                "input_variant": "baseline",
            },
            {
                "experiment": "mlp_trainonly_all_available",
                "method_variant": "set_thr045",
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.80,
                "mean_f1": 0.90,
                "input_variant": "baseline",
            },
            {
                "experiment": "ridge_trainonly_all_available",
                "method_variant": "set_thr050",
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.81,
                "mean_f1": 0.905,
                "input_variant": "baseline",
            },
        ]
    ).to_csv(table_dir / "candidate_set_weight_variant_fold_rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment": "mlp_trainonly_all_available",
                "dataset": "BF-C2DL-HSC",
                "selected_method_variant": "set_thr050",
                "validation_mean_iou": 0.83,
                "test_mean_iou": 0.82,
                "test_mean_f1": 0.91,
            }
        ]
    ).to_csv(
        table_dir
        / "mlp_trainonly_all_available_validation_selected_dataset_summary.csv",
        index=False,
    )

    selected = _candidate_set_weight_rows(
        root,
        experiment="mlp_trainonly_all_available",
        ablation_id="candidate_set_weight_mlp_trainonly",
        method_label="Candidate-set MLP-weighted object vote, train-only",
        phase2_role="candidate_set_learned_ensemble_trainonly",
        evidence_kind="candidate_set_weight_validation_selected",
        row_family="ensemble_qa",
        notes="test",
    )

    assert selected["selected_method"].tolist() == ["set_thr050"]
    assert selected["test_mean_iou"].tolist() == [0.82]
    assert selected["validation_mean_iou"].tolist() == [0.83]
    assert set(selected["row_family"]) == {"ensemble_qa"}
    assert set(selected["table_scope"]) == {"strict_qa_ensemble"}


def test_candidate_set_weight_rows_can_be_marked_diagnostic(
    tmp_path: Path,
) -> None:
    root = tmp_path
    table_dir = root / "candidate_set_weight_control_comparison"
    table_dir.mkdir()
    pd.DataFrame(
        [
            {
                "experiment": "mlp_trainonly_all_available",
                "method_variant": "set_thr050",
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.82,
                "mean_f1": 0.91,
            },
        ]
    ).to_csv(table_dir / "candidate_set_weight_variant_fold_rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment": "mlp_trainonly_all_available",
                "dataset": "BF-C2DL-HSC",
                "selected_method_variant": "set_thr050",
                "validation_mean_iou": 0.83,
            }
        ]
    ).to_csv(
        table_dir
        / "mlp_trainonly_all_available_validation_selected_dataset_summary.csv",
        index=False,
    )

    selected = _candidate_set_weight_rows(
        root,
        experiment="mlp_trainonly_all_available",
        ablation_id="candidate_set_weight_mlp_trainonly",
        method_label="Candidate-set MLP-weighted object vote, train-only",
        phase2_role="candidate_set_learned_ensemble_trainonly",
        evidence_kind="candidate_set_weight_validation_selected",
        row_family="ensemble_qa",
        notes="diagnostic",
        table_scope="diagnostic_qa_ensemble",
    )

    assert selected["selected_method"].tolist() == ["set_thr050"]
    assert set(selected["table_scope"]) == {"diagnostic_qa_ensemble"}


def test_candidate_set_weight_fixed_comparison_rows_are_strict(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run_dir = root / "candidate_set_weight_object_vote_mlp_trainonly_leave_target_video_out_thr050"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "selection_split": "test",
                "method_key": "ensemble_qa_candidate_set_mlp_leave_target_video_out_thr050",
                "mean_iou": 0.86,
                "mean_f1": 0.92,
            },
            {
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "selection_split": "validation",
                "method_key": "ensemble_qa_candidate_set_mlp_leave_target_video_out_thr050",
                "mean_iou": 0.87,
                "mean_f1": 0.93,
            },
        ]
    ).to_csv(run_dir / "qa_weighted_object_vote_comparison_rows.csv", index=False)

    rows = _candidate_set_weight_fixed_comparison_rows(
        root,
        rel_dir="candidate_set_weight_object_vote_mlp_trainonly_leave_target_video_out_thr050",
        ablation_id="candidate_set_weight_mlp_leave_target_video_out_thr050",
        method_label="Candidate-set MLP leave-target-video-out",
        phase2_role="candidate_set_learned_ensemble_leave_target_video_out",
        evidence_kind="candidate_set_weight_leave_target_video_out_fixed_thr050",
        row_family="ensemble_qa",
        notes="strict",
        selection_rule="fixed_threshold_0.50",
    )

    assert rows["test_mean_iou"].tolist() == [0.86]
    assert rows["selected_method"].tolist() == [
        "ensemble_qa_candidate_set_mlp_leave_target_video_out_thr050"
    ]
    assert set(rows["table_scope"]) == {"strict_qa_ensemble"}


def test_no_dataset_lodo_fixed_threshold_row_is_strict_qa_ensemble(
    tmp_path: Path,
) -> None:
    root = tmp_path
    table_dir = root / "candidate_set_weight_object_vote_selection_lodo_no_dataset"
    table_dir.mkdir()
    pd.DataFrame(
        [
            {
                "run_id": "set_thr055",
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.87,
                "mean_f1": 0.93,
            },
            {
                "run_id": "set_boundary075",
                "dataset": "BF-C2DL-HSC",
                "fold": "fold-1",
                "split": "test",
                "mean_iou": 0.86,
                "mean_f1": 0.92,
            },
        ]
    ).to_csv(table_dir / "object_vote_candidate_fold_rows.csv", index=False)

    fixed = _object_vote_fixed_run_rows(
        root,
        selection_rel_dir="candidate_set_weight_object_vote_selection_lodo_no_dataset",
        run_id="set_thr055",
        ablation_id="candidate_set_weight_mlp_lodo_no_dataset_fixed_thr055",
        method_label="Candidate-set MLP object vote, LODO no-dataset fixed threshold",
        phase2_role="candidate_set_cross_dataset_no_dataset_fixed_threshold",
        evidence_kind="candidate_set_weight_lodo_no_dataset_fixed_threshold",
        row_family="ensemble_qa",
        notes="test",
    )

    assert fixed["ablation_id"].tolist() == [
        "candidate_set_weight_mlp_lodo_no_dataset_fixed_thr055"
    ]
    assert fixed["selected_method"].tolist() == ["set_thr055"]
    assert fixed["test_mean_iou"].tolist() == [0.87]
    assert set(fixed["row_family"]) == {"ensemble_qa"}
    assert set(fixed["table_scope"]) == {"strict_qa_ensemble"}


def test_phase2_markdown_displays_table_scope(tmp_path: Path) -> None:
    output = tmp_path / "phase2.md"
    validation = pd.DataFrame(
        [
            {
                "method_label": "Diagnostic candidate-set MLP",
                "coverage": "all_available",
                "row_family": "ensemble_qa",
                "table_scope": "diagnostic_qa_ensemble",
                "datasets": 1,
                "mean_iou": 0.87,
                "mean_f1": 0.93,
            }
        ]
    )

    _write_markdown(
        output,
        validation_overall=validation,
        foldwise_best_overall=validation.iloc[0:0].copy(),
        coverage=pd.DataFrame(
            [
                {
                    "requested_axis": "candidate-set learned ENSEMBLE",
                    "status": "available",
                    "evidence": "candidate_set_weight_mlp_leave_target_video_out_thr050",
                    "coverage": "all_available",
                    "note": "strict row is leave-target-video.",
                }
            ]
        ),
        oracle_gaps=pd.DataFrame(),
    )

    markdown = output.read_text(encoding="utf-8")
    assert "table_scope" in markdown
    assert "diagnostic_qa_ensemble" in markdown
