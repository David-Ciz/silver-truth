import pandas as pd

from silver_truth.evaluation.cluster_pairwise_ranker import (
    ClusterPairwiseRankerConfig,
    score_with_cluster_pairwise_ranker,
    train_cluster_pairwise_ranker,
)


def test_cluster_pairwise_ranker_scores_cluster_winner() -> None:
    rows = []
    for idx in range(16):
        is_good = idx % 2 == 0
        rows.append(
            {
                "candidate_id": f"train_{idx}",
                "candidate_source": "good" if is_good else "bad",
                "image_id": f"train_img_{idx // 2}",
                "split": "train",
                "center_y": float(idx // 2),
                "center_x": float(idx // 2),
                "qa_score": 0.9 if is_good else 0.2,
                "qa_rank_in_image": 1.0 if is_good else 0.5,
                "fusion_support": 2.0,
                "fusion_support_rank_in_image": 1.0,
                "source_train_mean_iou": 0.8 if is_good else 0.4,
                "crop_area": 100.0,
                "true_iou": 0.9 if is_good else 0.3,
            }
        )
    rows.extend(
        [
            {
                "candidate_id": "test_bad",
                "candidate_source": "bad",
                "image_id": "test_img",
                "split": "test",
                "center_y": 10.0,
                "center_x": 10.0,
                "qa_score": 0.2,
                "qa_rank_in_image": 0.5,
                "fusion_support": 2.0,
                "fusion_support_rank_in_image": 1.0,
                "source_train_mean_iou": 0.4,
                "crop_area": 100.0,
                "true_iou": None,
            },
            {
                "candidate_id": "test_good",
                "candidate_source": "good",
                "image_id": "test_img",
                "split": "test",
                "center_y": 11.0,
                "center_x": 10.0,
                "qa_score": 0.9,
                "qa_rank_in_image": 1.0,
                "fusion_support": 2.0,
                "fusion_support_rank_in_image": 1.0,
                "source_train_mean_iou": 0.8,
                "crop_area": 100.0,
                "true_iou": None,
            },
        ]
    )
    features = pd.DataFrame(rows)

    payload = train_cluster_pairwise_ranker(
        features,
        config=ClusterPairwiseRankerConfig(use_source_identity=True),
    )
    scored = score_with_cluster_pairwise_ranker(features, model_payload=payload)
    test_scores = scored[scored["image_id"].eq("test_img")].set_index("candidate_id")

    assert payload["n_pairs"] > 0
    assert "ranker_cluster_temporal_rank" not in payload["feature_columns"]
    assert test_scores.loc["test_good", "pairwise_cluster_score"] > test_scores.loc[
        "test_bad", "pairwise_cluster_score"
    ]
    assert test_scores.loc["test_good", "pairwise_cluster_rank"] == 1.0


def test_cluster_pairwise_ranker_accepts_logistic_regularization() -> None:
    features = pd.DataFrame(
        [
            {
                "candidate_id": f"c{idx}",
                "candidate_source": "good" if idx % 2 == 0 else "bad",
                "image_id": f"img_{idx // 2}",
                "split": "train",
                "center_y": float(idx // 2),
                "center_x": 0.0,
                "qa_score": 0.9 if idx % 2 == 0 else 0.1,
                "fusion_support": 2.0,
                "crop_area": 100.0,
                "true_iou": 0.9 if idx % 2 == 0 else 0.2,
            }
            for idx in range(12)
        ]
    )

    payload = train_cluster_pairwise_ranker(
        features,
        config=ClusterPairwiseRankerConfig(
            use_source_identity=True,
            logistic_c=0.1,
        ),
    )

    assert payload["config"]["logistic_c"] == 0.1


def test_cluster_pairwise_ranker_can_blend_source_identity_model() -> None:
    features = pd.DataFrame(
        [
            {
                "candidate_id": f"c{idx}",
                "candidate_source": "good" if idx % 2 == 0 else "bad",
                "image_id": f"img_{idx // 2}",
                "split": "train",
                "center_y": float(idx // 2),
                "center_x": 0.0,
                "qa_score": 0.9 if idx % 2 == 0 else 0.1,
                "fusion_support": 2.0,
                "crop_area": 100.0,
                "true_iou": 0.9 if idx % 2 == 0 else 0.2,
            }
            for idx in range(12)
        ]
    )

    payload = train_cluster_pairwise_ranker(
        features,
        config=ClusterPairwiseRankerConfig(
            use_source_identity=True,
            source_identity_blend=0.5,
        ),
    )
    scored = score_with_cluster_pairwise_ranker(features, model_payload=payload)

    assert payload["config"]["source_identity_blend"] == 0.5
    assert payload["source_agnostic_model"] is not None
    assert scored["pairwise_cluster_score"].between(0.0, 1.0).all()
