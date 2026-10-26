from pathlib import Path

import pandas as pd

from silver_truth.evaluation.candidate_ranker import (
    CandidateRankerConfig,
    FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS,
    FULL_IMAGE_MASK_RANKER_FEATURE_COLUMNS,
    add_cluster_ranker_features,
    score_with_candidate_ranker,
    train_candidate_ranker,
)


def _feature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(24):
        quality = idx / 23
        split = "train" if idx < 16 else "validation"
        rows.append(
            {
                "image_id": f"{split}_{idx // 2}",
                "candidate_id": f"c{idx}",
                "candidate_source": "A" if idx % 2 == 0 else "B",
                "split": split,
                "center_y": float(idx // 2),
                "center_x": float(idx // 2),
                "fusion_support": 1 + quality,
                "fusion_support_rank_in_image": quality,
                "qa_score": quality,
                "qa_rank_in_image": quality,
                "crop_width": 32.0,
                "crop_height": 32.0,
                "crop_area": 1024.0,
                "temporal_min_center_dist_norm": 1.0 - quality,
                "source_temporal_min_center_dist_norm": 1.0 - quality,
                "temporal_support_score": quality,
                "source_temporal_support_score": quality,
                "source_train_mean_iou": 0.5 + (0.25 * quality),
                "dataset_train_mean_iou": 0.6,
                "true_iou": quality,
            }
        )
    rows.extend(
        [
            {
                "image_id": "test_0",
                "candidate_id": "low",
                "candidate_source": "A",
                "split": "test",
                "center_y": 10.0,
                "center_x": 10.0,
                "fusion_support": 1.0,
                "fusion_support_rank_in_image": 0.1,
                "qa_score": 0.1,
                "qa_rank_in_image": 0.1,
                "crop_width": 32.0,
                "crop_height": 32.0,
                "crop_area": 1024.0,
                "temporal_min_center_dist_norm": 0.9,
                "source_temporal_min_center_dist_norm": 0.9,
                "temporal_support_score": 0.1,
                "source_temporal_support_score": 0.1,
                "source_train_mean_iou": 0.5,
                "dataset_train_mean_iou": 0.6,
                "true_iou": None,
            },
            {
                "image_id": "test_0",
                "candidate_id": "high",
                "candidate_source": "B",
                "split": "test",
                "center_y": 11.0,
                "center_x": 10.0,
                "fusion_support": 2.0,
                "fusion_support_rank_in_image": 0.9,
                "qa_score": 0.9,
                "qa_rank_in_image": 0.9,
                "crop_width": 32.0,
                "crop_height": 32.0,
                "crop_area": 1024.0,
                "temporal_min_center_dist_norm": 0.1,
                "source_temporal_min_center_dist_norm": 0.1,
                "temporal_support_score": 0.9,
                "source_temporal_support_score": 0.9,
                "source_train_mean_iou": 0.75,
                "dataset_train_mean_iou": 0.6,
                "true_iou": None,
            },
        ]
    )
    return rows


def test_candidate_ranker_trains_calibrates_and_scores(tmp_path: Path) -> None:
    features = pd.DataFrame(_feature_rows())
    model_path = tmp_path / "candidate_ranker.joblib"

    payload = train_candidate_ranker(features, output_model=model_path)
    scored = score_with_candidate_ranker(features, model_payload=payload)

    test_scores = scored[scored["image_id"].eq("test_0")].set_index("candidate_id")
    assert model_path.exists()
    assert scored["candidate_ranker_score"].between(0.0, 1.0).all()
    assert test_scores.loc["high", "candidate_ranker_score"] > test_scores.loc[
        "low", "candidate_ranker_score"
    ]
    assert test_scores.loc["high", "candidate_ranker_rank_in_image"] == 1.0
    assert "candidate_ranker_rank_in_cluster" in scored.columns


def test_candidate_ranker_can_use_mlp_regressor() -> None:
    features = pd.DataFrame(_feature_rows())

    payload = train_candidate_ranker(
        features,
        config=CandidateRankerConfig(
            model_type="mlp",
            mlp_hidden_layer_sizes=(8,),
            mlp_max_iter=300,
        ),
    )
    scored = score_with_candidate_ranker(features, model_payload=payload)

    test_scores = scored[scored["image_id"].eq("test_0")].set_index("candidate_id")
    assert payload["config"]["model_type"] == "mlp"
    assert scored["candidate_ranker_score"].between(0.0, 1.0).all()
    assert test_scores.loc["high", "candidate_ranker_score"] > test_scores.loc[
        "low", "candidate_ranker_score"
    ]


def test_candidate_ranker_can_use_source_cluster_quality_prior() -> None:
    features = pd.DataFrame(_feature_rows())

    payload = train_candidate_ranker(
        features,
        config=CandidateRankerConfig(use_source_cluster_quality_prior=True),
    )
    scored = score_with_candidate_ranker(features, model_payload=payload)

    assert payload["source_cluster_quality_priors"]
    assert "source_cluster_quality_prior" in scored.columns


def test_add_cluster_ranker_features_adds_local_ranks() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "i1",
                "center_y": 10.0,
                "center_x": 10.0,
                "qa_score": 0.2,
                "source_train_mean_iou": 0.9,
                "source_temporal_min_center_dist_norm": 2.0,
                "source_temporal_support_score": 0.2,
            },
            {
                "image_id": "i1",
                "center_y": 11.0,
                "center_x": 10.0,
                "qa_score": 0.8,
                "source_train_mean_iou": 0.5,
                "source_temporal_min_center_dist_norm": 0.2,
                "source_temporal_support_score": 0.8,
            },
        ]
    )

    augmented = add_cluster_ranker_features(features, cluster_distance=4.0)

    assert augmented["ranker_cluster_size"].tolist() == [2, 2]
    assert augmented["ranker_cluster_qa_rank"].tolist() == [0.5, 1.0]
    assert augmented["ranker_cluster_source_prior_rank"].tolist() == [1.0, 0.5]
    assert augmented["ranker_cluster_temporal_rank"].tolist() == [0.5, 1.0]
    assert augmented["ranker_cluster_temporal_support_rank"].tolist() == [0.5, 1.0]


def test_add_cluster_ranker_features_adds_rect_image_relative_features() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "i1",
                "center_y": 10.0,
                "center_x": 10.0,
                "rect_raw_contrast_norm": 1.0,
                "rect_raw_boundary_contrast": 5.0,
                "rect_mask_extent": 0.4,
                "rect_mask_area_pixels": 50.0,
            },
            {
                "image_id": "i1",
                "center_y": 11.0,
                "center_x": 10.0,
                "rect_raw_contrast_norm": 2.0,
                "rect_raw_boundary_contrast": 3.0,
                "rect_mask_extent": 0.8,
                "rect_mask_area_pixels": 100.0,
            },
        ]
    )

    augmented = add_cluster_ranker_features(features, cluster_distance=4.0)

    assert augmented["ranker_cluster_rect_raw_contrast_norm_rank"].tolist() == [
        0.5,
        1.0,
    ]
    assert augmented["ranker_cluster_rect_boundary_contrast_rank"].tolist() == [
        1.0,
        0.5,
    ]
    assert augmented["ranker_cluster_rect_mask_extent_rank"].tolist() == [0.5, 1.0]
    assert augmented["rect_mask_area_abs_log_ratio_to_cluster_median"].notna().all()


def test_add_cluster_ranker_features_adds_full_image_relative_features() -> None:
    features = pd.DataFrame(
        [
            {
                "image_id": "i1",
                "center_y": 10.0,
                "center_x": 10.0,
                "full_raw_contrast_norm": 1.0,
                "full_raw_boundary_contrast": 5.0,
                "full_mask_extent": 0.4,
                "full_mask_area_pixels": 50.0,
            },
            {
                "image_id": "i1",
                "center_y": 11.0,
                "center_x": 10.0,
                "full_raw_contrast_norm": 2.0,
                "full_raw_boundary_contrast": 3.0,
                "full_mask_extent": 0.8,
                "full_mask_area_pixels": 100.0,
            },
        ]
    )

    augmented = add_cluster_ranker_features(features, cluster_distance=4.0)

    assert augmented["ranker_cluster_full_raw_contrast_norm_rank"].tolist() == [
        0.5,
        1.0,
    ]
    assert augmented["ranker_cluster_full_boundary_contrast_rank"].tolist() == [
        1.0,
        0.5,
    ]
    assert augmented["ranker_cluster_full_mask_extent_rank"].tolist() == [0.5, 1.0]
    assert augmented["full_mask_area_abs_log_ratio_to_cluster_median"].notna().all()


def test_full_image_boundary_noise_ranker_columns_are_separate() -> None:
    assert "full_raw_contrast_to_noise" not in FULL_IMAGE_MASK_RANKER_FEATURE_COLUMNS
    assert "full_raw_contrast_to_noise" in FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS
    assert (
        "full_raw_boundary_gradient_mean"
        in FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS
    )
    assert (
        "full_raw_boundary_gradient_contrast"
        in FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS
    )
    assert "full_mask_hole_fraction" in FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS
    assert "full_mask_eccentricity" in FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS
