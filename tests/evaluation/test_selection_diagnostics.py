import pandas as pd

from silver_truth.evaluation.selection_diagnostics import (
    build_selection_diagnostics,
    check_duplicate_sources,
)


def test_build_selection_diagnostics_reports_cluster_oracle_gap() -> None:
    features = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "image_id": "01_t001",
                "split": "test",
                "candidate_source": "A",
                "candidate_source_raw": "A",
                "center_y": 10,
                "center_x": 10,
                "true_iou": 0.9,
                "candidate_ranker_score": 0.2,
                "qa_score": 0.2,
            },
            {
                "candidate_id": "b",
                "image_id": "01_t001",
                "split": "test",
                "candidate_source": "B",
                "candidate_source_raw": "B",
                "center_y": 11,
                "center_x": 10,
                "true_iou": 0.5,
                "candidate_ranker_score": 0.8,
                "qa_score": 0.8,
            },
            {
                "candidate_id": "c",
                "image_id": "01_t001",
                "split": "test",
                "candidate_source": "C",
                "candidate_source_raw": "C",
                "center_y": 30,
                "center_x": 30,
                "true_iou": 0.7,
                "candidate_ranker_score": 0.7,
                "qa_score": 0.7,
            },
        ]
    )
    selected = pd.DataFrame({"candidate_id": ["b", "c"]})

    diagnostics = build_selection_diagnostics(
        features, selected, split="test", cluster_distance=4.0
    )

    clusters = diagnostics["cluster_oracle_gap"].sort_values("oracle_gap")
    assert len(clusters) == 2
    assert clusters["selected_count"].sum() == 2
    assert clusters["oracle_gap"].max() == 0.4


def test_check_duplicate_sources_compares_raw_alias_rows() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "01_t001",
                "label": "1",
                "candidate_source_raw": "BGU-IL (5)",
                "center_y": 10,
                "center_x": 20,
                "crop_y_start": 1,
                "crop_y_end": 2,
                "crop_x_start": 3,
                "crop_x_end": 4,
                "fusion_support": 5,
                "qa_score": 0.7,
                "true_iou": 0.8,
            },
            {
                "image_id": "01_t001",
                "label": "1",
                "candidate_source_raw": "BGU-IL-5",
                "center_y": 10,
                "center_x": 20,
                "crop_y_start": 1,
                "crop_y_end": 2,
                "crop_x_start": 3,
                "crop_x_end": 4,
                "fusion_support": 5,
                "qa_score": 0.7,
                "true_iou": 0.8,
            },
        ]
    )

    check = check_duplicate_sources(
        features, source_pairs=[("BGU-IL (5)", "BGU-IL-5")]
    )

    assert check.loc[0, "paired_groups"] == 1
    assert check.loc[0, "max_abs_diff_true_iou"] == 0.0
