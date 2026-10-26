from pathlib import Path

import pandas as pd

from silver_truth.evaluation.candidate_set_weight import (
    CandidateSetWeightConfig,
    _deep_sets_clusters,
    _deep_sets_oversize_area_values,
    add_candidate_set_context_features,
    add_candidate_set_relative_features,
    score_with_candidate_set_weight_predictor,
    train_candidate_set_weight_predictor,
    write_leave_one_dataset_out_weighted_features,
    write_leave_target_video_out_weighted_features,
    write_candidate_set_weighted_features,
)
from scripts.run_candidate_set_weight_predictor import _config_from_args
from scripts.qa_pruned_ensemble.run_pruned_candidate_set_weight import (
    EXPLICIT_QA_FEATURE_COLUMNS,
    _config_from_args as _pruned_config_from_args,
)


def _feature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(30):
        split = "train" if idx < 18 else "validation"
        quality = 0.9 if idx % 2 else 0.2
        rows.append(
            {
                "dataset": "toy",
                "image_id": f"{split}_{idx // 2}",
                "candidate_id": f"candidate_{idx}",
                "candidate_source": "strong" if idx % 2 else "weak",
                "split": split,
                "center_y": float(idx // 2),
                "center_x": 10.0,
                "qa_score": quality,
                "qa_rank_in_image": quality,
                "fusion_support": 1.0 + quality,
                "fusion_support_rank_in_image": quality,
                "source_train_mean_iou": quality,
                "dataset_train_mean_iou": 0.55,
                "pairwise_cluster_score": quality,
                "pairwise_cluster_rank": quality,
                "local_image_support_score": quality,
                "true_iou": quality,
            }
        )
    rows.extend(
        [
            {
                "dataset": "toy",
                "image_id": "test_0",
                "candidate_id": "test_low",
                "candidate_source": "weak",
                "split": "test",
                "center_y": 1.0,
                "center_x": 1.0,
                "qa_score": 0.1,
                "qa_rank_in_image": 0.1,
                "fusion_support": 1.1,
                "fusion_support_rank_in_image": 0.1,
                "source_train_mean_iou": 0.2,
                "dataset_train_mean_iou": 0.55,
                "pairwise_cluster_score": 0.1,
                "pairwise_cluster_rank": 0.1,
                "local_image_support_score": 0.1,
                "true_iou": None,
            },
            {
                "dataset": "toy",
                "image_id": "test_0",
                "candidate_id": "test_high",
                "candidate_source": "strong",
                "split": "test",
                "center_y": 2.0,
                "center_x": 1.0,
                "qa_score": 0.9,
                "qa_rank_in_image": 0.9,
                "fusion_support": 1.9,
                "fusion_support_rank_in_image": 0.9,
                "source_train_mean_iou": 0.9,
                "dataset_train_mean_iou": 0.55,
                "pairwise_cluster_score": 0.9,
                "pairwise_cluster_rank": 0.9,
                "local_image_support_score": 0.9,
                "true_iou": None,
            },
        ]
    )
    return rows


def test_candidate_set_weight_predictor_scores_cluster_winner() -> None:
    features = pd.DataFrame(_feature_rows())

    payload = train_candidate_set_weight_predictor(
        features,
        config=CandidateSetWeightConfig(
            train_splits=("train",),
            model_type="ridge",
            categorical_columns=(),
            min_training_rows=10,
        ),
    )
    scored = score_with_candidate_set_weight_predictor(features, model_payload=payload)

    test_scores = scored[scored["image_id"].eq("test_0")].set_index("candidate_id")
    assert scored["candidate_set_weight_score"].between(0.0, 1.0).all()
    assert test_scores.loc["test_high", "candidate_set_weight_score"] > test_scores.loc[
        "test_low", "candidate_set_weight_score"
    ]
    assert test_scores.loc["test_high", "candidate_set_weight_rank"] == 1.0


def test_candidate_set_relative_features_include_rank_and_gap() -> None:
    features = pd.DataFrame(_feature_rows()[-2:])

    augmented = add_candidate_set_relative_features(
        features,
        config=CandidateSetWeightConfig(categorical_columns=()),
    )

    low = augmented.set_index("candidate_id").loc["test_low"]
    high = augmented.set_index("candidate_id").loc["test_high"]
    assert high["candidate_set_rank__qa_score"] == 1.0
    assert low["candidate_set_gap_to_best__qa_score"] > 0.0


def test_candidate_set_context_features_include_cluster_summaries() -> None:
    features = pd.DataFrame(_feature_rows()[-2:])
    relative = add_candidate_set_relative_features(features)

    augmented = add_candidate_set_context_features(relative)

    low = augmented.set_index("candidate_id").loc["test_low"]
    assert low["candidate_set_context__cluster_size"] == 2
    assert low["candidate_set_context__mean__qa_score"] == 0.5
    assert low["candidate_set_context__max__qa_score"] == 0.9


def test_set_mlp_candidate_set_weight_predictor_records_context_columns() -> None:
    features = pd.DataFrame(_feature_rows())

    payload = train_candidate_set_weight_predictor(
        features,
        config=CandidateSetWeightConfig(
            model_type="set_mlp",
            categorical_columns=(),
            mlp_hidden_layer_sizes=(8,),
            mlp_max_iter=50,
            min_training_rows=10,
        ),
    )
    scored = score_with_candidate_set_weight_predictor(features, model_payload=payload)

    assert payload["config"]["model_type"] == "set_mlp"
    assert any(
        column.startswith("candidate_set_context__")
        for column in payload["feature_columns"]
    )
    assert "candidate_set_context__cluster_size" in scored.columns
    assert "candidate_set_weight_score" in scored.columns


def test_deep_sets_candidate_set_weight_predictor_is_order_invariant() -> None:
    features = pd.DataFrame(_feature_rows())

    payload = train_candidate_set_weight_predictor(
        features,
        config=CandidateSetWeightConfig(
            model_type="deep_sets",
            categorical_columns=(),
            deep_sets_hidden_dim=8,
            deep_sets_epochs=4,
            deep_sets_batch_clusters=4,
            deep_sets_device="cpu",
            min_training_rows=10,
        ),
    )
    scored = score_with_candidate_set_weight_predictor(features, model_payload=payload)
    shuffled = features.sample(frac=1.0, random_state=7).reset_index(drop=True)
    rescored = score_with_candidate_set_weight_predictor(
        shuffled, model_payload=payload
    )

    left = scored.set_index("candidate_id")["candidate_set_weight_score"].sort_index()
    right = rescored.set_index("candidate_id")["candidate_set_weight_score"].sort_index()
    assert payload["model_type"] == "deep_sets"
    assert payload["train_clusters"] > 0
    assert (left - right).abs().max() < 1e-6


def test_deep_sets_candidate_set_weight_predictor_scores_cluster_winner() -> None:
    features = pd.DataFrame(_feature_rows())

    payload = train_candidate_set_weight_predictor(
        features,
        config=CandidateSetWeightConfig(
            model_type="deep_sets",
            categorical_columns=(),
            deep_sets_hidden_dim=8,
            deep_sets_epochs=20,
            deep_sets_batch_clusters=4,
            deep_sets_device="cpu",
            min_training_rows=10,
        ),
    )
    scored = score_with_candidate_set_weight_predictor(features, model_payload=payload)

    test_scores = scored[scored["image_id"].eq("test_0")].set_index("candidate_id")
    assert scored["candidate_set_weight_score"].between(0.0, 1.0).all()
    assert test_scores.loc["test_high", "candidate_set_weight_score"] > test_scores.loc[
        "test_low", "candidate_set_weight_score"
    ]


def test_write_candidate_set_weighted_features_mirrors_relative_paths(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "features"
    feature_path = feature_root / "dataset" / "fold-1_paper_ready_pairwise_ranked.parquet"
    feature_path.parent.mkdir(parents=True)
    pd.DataFrame(_feature_rows()).to_parquet(feature_path, index=False)

    summary = write_candidate_set_weighted_features(
        [feature_path],
        output_root=tmp_path / "weighted",
        relative_root=feature_root,
        config=CandidateSetWeightConfig(
            model_type="ridge",
            categorical_columns=(),
            min_training_rows=10,
        ),
    )

    output_path = tmp_path / "weighted" / "dataset" / "fold-1_paper_ready_pairwise_ranked.parquet"
    scored = pd.read_parquet(output_path)
    assert summary.loc[0, "output_path"] == str(output_path)
    assert "candidate_set_weight_score" in scored.columns


def test_write_leave_one_dataset_out_weighted_features_trains_other_datasets(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "features"
    paths = []
    for dataset in ("A", "B"):
        feature_path = (
            feature_root
            / dataset
            / "fold-1_paper_ready_pairwise_ranked.parquet"
        )
        feature_path.parent.mkdir(parents=True)
        rows = []
        for row in _feature_rows():
            copied = dict(row)
            copied["dataset"] = dataset
            copied["candidate_id"] = f"{dataset}_{copied['candidate_id']}"
            rows.append(copied)
        pd.DataFrame(rows).to_parquet(feature_path, index=False)
        paths.append(feature_path)

    summary = write_leave_one_dataset_out_weighted_features(
        paths,
        output_root=tmp_path / "weighted",
        relative_root=feature_root,
        output_model_root=tmp_path / "models",
        config=CandidateSetWeightConfig(
            train_splits=("train",),
            model_type="ridge",
            categorical_columns=(),
            min_training_rows=10,
        ),
    )

    assert set(summary["held_out_dataset"]) == {"A", "B"}
    assert (tmp_path / "models" / "A_candidate_set_weight_predictor.joblib").exists()
    assert (tmp_path / "models" / "B_candidate_set_weight_predictor.joblib").exists()
    for dataset in ("A", "B"):
        output_path = (
            tmp_path
            / "weighted"
            / dataset
            / "fold-1_paper_ready_pairwise_ranked.parquet"
        )
        scored = pd.read_parquet(output_path)
        assert "candidate_set_weight_score" in scored.columns


def test_write_leave_target_video_out_weighted_features_excludes_target_campaign(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "features"
    paths = []
    for fold, train_campaign, test_campaign in (
        ("fold-1", "01", "02"),
        ("fold-2", "02", "01"),
    ):
        feature_path = (
            feature_root
            / "toy"
            / "sz"
            / "variant"
            / f"{fold}_paper_ready_pairwise_ranked.parquet"
        )
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, row in enumerate(_feature_rows()):
            copied = dict(row)
            copied["dataset"] = "toy"
            copied["fold"] = fold
            copied["campaign_number"] = (
                test_campaign if copied["split"] == "test" else train_campaign
            )
            copied["candidate_id"] = f"{fold}_{idx}_{copied['candidate_id']}"
            rows.append(copied)
        pd.DataFrame(rows).to_parquet(feature_path, index=False)
        paths.append(feature_path)

    summary = write_leave_target_video_out_weighted_features(
        paths,
        output_root=tmp_path / "weighted",
        relative_root=feature_root,
        output_model_root=tmp_path / "models",
        config=CandidateSetWeightConfig(
            train_splits=("train",),
            model_type="ridge",
            categorical_columns=(),
            min_training_rows=10,
        ),
    )

    assert set(summary["held_out_campaign"]) == {"01", "02"}
    assert set(summary["train_rows"]) == {18}
    assert (
        tmp_path / "models" / "toy_fold-1_leave_target_video_out.joblib"
    ).exists()
    assert (
        tmp_path / "models" / "toy_fold-2_leave_target_video_out.joblib"
    ).exists()
    for fold in ("fold-1", "fold-2"):
        output_path = (
            tmp_path
            / "weighted"
            / "toy"
            / "sz"
            / "variant"
            / f"{fold}_paper_ready_pairwise_ranked.parquet"
        )
        scored = pd.read_parquet(output_path)
        assert "candidate_set_weight_score" in scored.columns


def test_candidate_set_weight_cli_accepts_custom_categorical_columns() -> None:
    class Args:
        train_split = ["train"]
        model_type = "ridge"
        categorical_column = ["candidate_source"]
        mlp_hidden_layer_size = [8]
        mlp_alpha = 0.01
        mlp_max_iter = 25
        target_temperature = 0.2
        min_score = 0.05
        random_state = 5
        categorical_dropout_column = ["dataset"]
        categorical_dropout_probability = 0.25
        dataset_dropout_prob = None

    config = _config_from_args(Args())

    assert config.train_splits == ("train",)
    assert config.model_type == "ridge"
    assert config.categorical_columns == ("candidate_source",)
    assert config.categorical_dropout_columns == ("dataset",)
    assert config.categorical_dropout_probability == 0.25


def test_candidate_set_weight_cli_accepts_set_mlp_dataset_dropout_alias() -> None:
    class Args:
        train_split = ["train"]
        model_type = "set_mlp"
        categorical_column = ["dataset", "candidate_source"]
        mlp_hidden_layer_size = [8]
        mlp_alpha = 0.01
        mlp_max_iter = 25
        target_temperature = 0.2
        min_score = 0.05
        random_state = 5
        categorical_dropout_column = None
        categorical_dropout_probability = 0.0
        dataset_dropout_prob = 0.4

    config = _config_from_args(Args())

    assert config.model_type == "set_mlp"
    assert config.categorical_dropout_columns == ("dataset",)
    assert config.categorical_dropout_probability == 0.4


def test_candidate_set_weight_cli_accepts_deep_sets_options() -> None:
    class Args:
        train_split = ["train"]
        model_type = "deep_sets"
        categorical_column = ["dataset", "candidate_source"]
        mlp_hidden_layer_size = [8]
        mlp_alpha = 0.01
        mlp_max_iter = 25
        target_temperature = 0.2
        min_score = 0.05
        random_state = 5
        categorical_dropout_column = None
        categorical_dropout_probability = 0.0
        dataset_dropout_prob = None
        deep_sets_hidden_dim = 12
        deep_sets_epochs = 9
        deep_sets_learning_rate = 0.002
        deep_sets_weight_decay = 0.003
        deep_sets_batch_clusters = 5
        deep_sets_device = "cpu"
        deep_sets_bad_mass_penalty_weight = 0.7
        deep_sets_bad_mass_margin = 0.12
        deep_sets_oversize_bad_mass_penalty_weight = 0.9
        deep_sets_oversize_bad_mass_margin = 0.15
        deep_sets_oversize_area_column = "rect_mask_area_log_ratio_to_cluster_median"
        deep_sets_oversize_area_log_ratio_threshold = 0.35

    config = _config_from_args(Args())

    assert config.model_type == "deep_sets"
    assert config.deep_sets_hidden_dim == 12
    assert config.deep_sets_epochs == 9
    assert config.deep_sets_learning_rate == 0.002
    assert config.deep_sets_weight_decay == 0.003
    assert config.deep_sets_batch_clusters == 5
    assert config.deep_sets_device == "cpu"
    assert config.deep_sets_bad_mass_penalty_weight == 0.7
    assert config.deep_sets_bad_mass_margin == 0.12
    assert config.deep_sets_oversize_bad_mass_penalty_weight == 0.9
    assert config.deep_sets_oversize_bad_mass_margin == 0.15
    assert (
        config.deep_sets_oversize_area_column
        == "rect_mask_area_log_ratio_to_cluster_median"
    )
    assert config.deep_sets_oversize_area_log_ratio_threshold == 0.35


def test_pruned_candidate_set_weight_cli_can_exclude_explicit_qa_features() -> None:
    class Args:
        train_split = ["train"]
        model_type = "deep_sets"
        categorical_column = ["dataset", "candidate_source"]
        mlp_hidden_layer_size = [8]
        mlp_alpha = 0.01
        mlp_max_iter = 25
        target_temperature = 0.2
        min_score = 0.05
        random_state = 5
        deep_sets_hidden_dim = 12
        deep_sets_epochs = 9
        deep_sets_learning_rate = 0.002
        deep_sets_weight_decay = 0.003
        deep_sets_batch_clusters = 5
        deep_sets_device = "cpu"
        deep_sets_bad_mass_penalty_weight = 0.7
        deep_sets_bad_mass_margin = 0.12
        exclude_feature_column = None
        exclude_explicit_qa_features = True

    config = _pruned_config_from_args(Args())

    assert not set(EXPLICIT_QA_FEATURE_COLUMNS).intersection(config.feature_columns)


def test_deep_sets_oversize_bad_mass_marks_only_bad_oversized_candidates() -> None:
    matrix = pd.DataFrame({"feature": [0.9, 0.3, 0.2]}).to_numpy(dtype="float32")
    target = pd.Series([1.0, 0.05, 0.02]).to_numpy(dtype="float32")
    true_iou = pd.Series([0.9, 0.5, 0.4]).to_numpy(dtype="float32")
    oversize_area = pd.Series([0.10, 0.35, -0.20]).to_numpy(dtype="float32")
    cluster_ids = pd.Series(["cluster", "cluster", "cluster"]).to_numpy()

    clusters = _deep_sets_clusters(
        matrix,
        target,
        true_iou,
        oversize_area,
        cluster_ids,
        bad_mass_margin=0.10,
        oversize_bad_mass_margin=0.10,
        oversize_area_log_ratio_threshold=0.25,
    )

    _, _, bad, oversize_bad = clusters[0]
    assert bad.tolist() == [False, True, True]
    assert oversize_bad.tolist() == [False, True, False]


def test_deep_sets_oversize_area_values_missing_column_is_nan() -> None:
    values = _deep_sets_oversize_area_values(
        pd.DataFrame({"other": [1.0, 2.0]}),
        config=CandidateSetWeightConfig(
            deep_sets_oversize_area_column="missing_area_feature"
        ),
    )

    assert pd.isna(values).all()


def test_candidate_set_weight_predictor_records_categorical_dropout_config() -> None:
    features = pd.DataFrame(_feature_rows())

    payload = train_candidate_set_weight_predictor(
        features,
        config=CandidateSetWeightConfig(
            model_type="ridge",
            categorical_columns=("dataset", "candidate_source"),
            categorical_dropout_columns=("dataset",),
            categorical_dropout_probability=1.0,
            min_training_rows=10,
        ),
    )
    scored = score_with_candidate_set_weight_predictor(features, model_payload=payload)

    assert payload["config"]["categorical_dropout_columns"] == ("dataset",)
    assert payload["config"]["categorical_dropout_probability"] == 1.0
    assert "candidate_set_weight_score" in scored.columns
