import pandas as pd

from silver_truth.evaluation.soft_hybrid import (
    add_local_image_support_score,
    run_soft_hybrid_selection,
    score_candidates,
)


def test_score_candidates_uses_support_qa_and_temporal_features() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "02_t001",
                "candidate_id": "a",
                "fusion_support_rank_in_image": 1.0,
                "qa_rank_in_image": 0.5,
                "source_temporal_min_center_dist_norm": 0.1,
                "crop_area": 100.0,
                "source_train_mean_iou": 0.8,
                "dataset_train_mean_iou": 0.7,
            },
            {
                "image_id": "02_t001",
                "candidate_id": "b",
                "fusion_support_rank_in_image": 0.2,
                "qa_rank_in_image": 1.0,
                "source_temporal_min_center_dist_norm": 3.0,
                "crop_area": 100.0,
                "source_train_mean_iou": 0.8,
                "dataset_train_mean_iou": 0.7,
            },
        ]
    )

    scored = score_candidates(features)

    assert scored.loc[0, "soft_hybrid_score"] > scored.loc[1, "soft_hybrid_score"]


def test_score_candidates_prefers_temporal_support_score_when_present() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "02_t001",
                "candidate_id": "a",
                "fusion_support_rank_in_image": 0.5,
                "qa_rank_in_image": 0.5,
                "source_temporal_support_score": 0.9,
                "source_temporal_min_center_dist_norm": 10.0,
                "crop_area": 100.0,
                "source_train_mean_iou": 0.5,
                "dataset_train_mean_iou": 0.5,
            },
            {
                "image_id": "02_t001",
                "candidate_id": "b",
                "fusion_support_rank_in_image": 0.5,
                "qa_rank_in_image": 0.5,
                "source_temporal_support_score": 0.1,
                "source_temporal_min_center_dist_norm": 0.1,
                "crop_area": 100.0,
                "source_train_mean_iou": 0.5,
                "dataset_train_mean_iou": 0.5,
            },
        ]
    )

    scored = score_candidates(features)

    assert scored.loc[0, "soft_temporal"] > scored.loc[1, "soft_temporal"]


def test_score_candidates_prefers_calibrated_ranker_when_present() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "02_t001",
                "candidate_id": "a",
                "fusion_support_rank_in_image": 0.5,
                "qa_rank_in_image": 0.1,
                "candidate_ranker_rank_in_image": 1.0,
                "source_temporal_min_center_dist_norm": 1.0,
                "crop_area": 100.0,
                "source_train_mean_iou": 0.5,
                "dataset_train_mean_iou": 0.5,
            },
            {
                "image_id": "02_t001",
                "candidate_id": "b",
                "fusion_support_rank_in_image": 0.5,
                "qa_rank_in_image": 1.0,
                "candidate_ranker_rank_in_image": 0.1,
                "source_temporal_min_center_dist_norm": 1.0,
                "crop_area": 100.0,
                "source_train_mean_iou": 0.5,
                "dataset_train_mean_iou": 0.5,
            },
        ]
    )

    scored = score_candidates(features)

    assert scored.loc[0, "soft_qa"] > scored.loc[1, "soft_qa"]


def test_score_candidates_uses_local_image_support_score_when_present() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "02_t001",
                "candidate_id": "a",
                "fusion_support_rank_in_image": 0.5,
                "qa_rank_in_image": 0.5,
                "source_temporal_min_center_dist_norm": 1.0,
                "crop_area": 100.0,
                "source_train_mean_iou": 0.5,
                "dataset_train_mean_iou": 0.5,
                "local_image_support_score": 1.0,
            },
            {
                "image_id": "02_t001",
                "candidate_id": "b",
                "fusion_support_rank_in_image": 0.5,
                "qa_rank_in_image": 0.5,
                "source_temporal_min_center_dist_norm": 1.0,
                "crop_area": 100.0,
                "source_train_mean_iou": 0.5,
                "dataset_train_mean_iou": 0.5,
                "local_image_support_score": 0.0,
            },
        ]
    )

    scored = score_candidates(features)

    assert scored.loc[0, "soft_local_image_support"] == 1.0
    assert scored.loc[1, "soft_local_image_support"] == 0.0
    assert scored.loc[0, "soft_hybrid_score"] > scored.loc[1, "soft_hybrid_score"]


def test_add_local_image_support_score_averages_available_cluster_contrast_ranks() -> None:
    features = pd.DataFrame(
        [
            {
                "ranker_cluster_full_boundary_contrast_rank": 1.0,
                "ranker_cluster_rect_boundary_contrast_rank": 0.5,
            },
            {
                "ranker_cluster_full_boundary_contrast_rank": 0.0,
                "ranker_cluster_rect_boundary_contrast_rank": 0.5,
            },
        ]
    )

    augmented = add_local_image_support_score(features)

    assert augmented["local_image_support_score"].tolist() == [0.75, 0.25]


def test_add_local_image_support_score_supports_boundary_modes() -> None:
    features = pd.DataFrame(
        [
            {
                "ranker_cluster_full_boundary_contrast_rank": 1.0,
                "ranker_cluster_rect_boundary_contrast_rank": 0.5,
            },
            {
                "ranker_cluster_full_boundary_contrast_rank": 0.0,
                "ranker_cluster_rect_boundary_contrast_rank": 0.5,
            },
        ]
    )

    full = add_local_image_support_score(features, mode="full_boundary")
    rect = add_local_image_support_score(features, mode="rect_boundary")
    boundary_min = add_local_image_support_score(features, mode="boundary_min")

    assert full["local_image_support_score"].tolist() == [1.0, 0.0]
    assert rect["local_image_support_score"].tolist() == [0.5, 0.5]
    assert boundary_min["local_image_support_score"].tolist() == [0.5, 0.0]


def test_run_soft_hybrid_selection_suppresses_overlapping_lower_score_candidates() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "02_t001",
                "candidate_id": "a",
                "crop_y_start": 0,
                "crop_y_end": 10,
                "crop_x_start": 0,
                "crop_x_end": 10,
                "center_y": 5,
                "center_x": 5,
                "crop_area": 100.0,
                "fusion_support": 5,
                "qa_score": 0.6,
                "fusion_support_rank_in_image": 1.0,
                "qa_rank_in_image": 0.5,
                "source_temporal_min_center_dist_norm": 0.1,
                "source_train_mean_iou": 0.8,
                "dataset_train_mean_iou": 0.7,
            },
            {
                "image_id": "02_t001",
                "candidate_id": "b",
                "crop_y_start": 1,
                "crop_y_end": 11,
                "crop_x_start": 1,
                "crop_x_end": 11,
                "center_y": 6,
                "center_x": 6,
                "crop_area": 100.0,
                "fusion_support": 2,
                "qa_score": 0.9,
                "fusion_support_rank_in_image": 0.5,
                "qa_rank_in_image": 1.0,
                "source_temporal_min_center_dist_norm": 2.0,
                "source_train_mean_iou": 0.8,
                "dataset_train_mean_iou": 0.7,
            },
            {
                "image_id": "02_t001",
                "candidate_id": "c",
                "crop_y_start": 30,
                "crop_y_end": 40,
                "crop_x_start": 30,
                "crop_x_end": 40,
                "center_y": 35,
                "center_x": 35,
                "crop_area": 100.0,
                "fusion_support": 1,
                "qa_score": 0.4,
                "fusion_support_rank_in_image": 0.1,
                "qa_rank_in_image": 0.1,
                "source_temporal_min_center_dist_norm": 1.0,
                "source_train_mean_iou": 0.6,
                "dataset_train_mean_iou": 0.7,
            },
        ]
    )

    selected = run_soft_hybrid_selection(features, max_bbox_iou=0.3)

    assert selected["candidate_id"].tolist() == ["a", "c"]


def test_run_soft_hybrid_selection_keeps_nearby_cells_with_overlapping_crop_boxes() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "02_t001",
                "candidate_id": "a",
                "crop_y_start": 0,
                "crop_y_end": 64,
                "crop_x_start": 0,
                "crop_x_end": 64,
                "center_y": 20,
                "center_x": 20,
                "crop_area": 4096.0,
                "fusion_support": 5,
                "qa_score": 0.8,
                "fusion_support_rank_in_image": 1.0,
                "qa_rank_in_image": 1.0,
                "source_temporal_min_center_dist_norm": 0.1,
                "source_train_mean_iou": 0.8,
                "dataset_train_mean_iou": 0.7,
            },
            {
                "image_id": "02_t001",
                "candidate_id": "b",
                "crop_y_start": 10,
                "crop_y_end": 74,
                "crop_x_start": 10,
                "crop_x_end": 74,
                "center_y": 35,
                "center_x": 35,
                "crop_area": 4096.0,
                "fusion_support": 4,
                "qa_score": 0.7,
                "fusion_support_rank_in_image": 0.8,
                "qa_rank_in_image": 0.8,
                "source_temporal_min_center_dist_norm": 0.1,
                "source_train_mean_iou": 0.8,
                "dataset_train_mean_iou": 0.7,
            },
        ]
    )

    selected = run_soft_hybrid_selection(features)

    assert selected["candidate_id"].tolist() == ["a", "b"]
