from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_RANKER_FEATURE_COLUMNS = (
    "fusion_support",
    "fusion_support_rank_in_image",
    "qa_score",
    "qa_rank_in_image",
    "ranker_cluster_size",
    "ranker_cluster_qa_rank",
    "ranker_cluster_source_prior_rank",
    "ranker_cluster_temporal_rank",
    "crop_width",
    "crop_height",
    "crop_area",
    "temporal_min_center_dist_norm",
    "source_temporal_min_center_dist_norm",
    "source_train_mean_iou",
    "dataset_train_mean_iou",
)

TEMPORAL_SUPPORT_RANKER_FEATURE_COLUMNS = (
    "temporal_support_score",
    "source_temporal_support_score",
    "ranker_cluster_temporal_support_rank",
)

IMAGE_MASK_RANKER_FEATURE_COLUMNS = (
    "mask_area_pixels",
    "mask_area_fraction",
    "mask_bbox_area",
    "mask_extent",
    "mask_perimeter_area_ratio",
    "mask_component_count",
    "mask_border_touch_fraction",
    "mask_centroid_offset_norm",
    "raw_contrast",
    "raw_contrast_norm",
    "raw_boundary_contrast",
)

RECT_IMAGE_MASK_RANKER_FEATURE_COLUMNS = (
    "rect_mask_area_pixels",
    "rect_mask_area_fraction",
    "rect_mask_bbox_aspect_ratio",
    "rect_mask_bbox_area",
    "rect_mask_extent",
    "rect_mask_perimeter_area_ratio",
    "rect_mask_component_count",
    "rect_mask_border_touch_fraction",
    "rect_mask_centroid_offset_norm",
    "rect_raw_contrast",
    "rect_raw_contrast_norm",
    "rect_raw_boundary_contrast",
    "ranker_cluster_rect_raw_contrast_norm_rank",
    "ranker_cluster_rect_boundary_contrast_rank",
    "ranker_cluster_rect_mask_extent_rank",
    "rect_mask_area_log_ratio_to_cluster_median",
    "rect_mask_area_abs_log_ratio_to_cluster_median",
)

CROP_GEOMETRY_RANKER_FEATURE_COLUMNS = (
    "crop_overflow_top",
    "crop_overflow_bottom",
    "crop_overflow_left",
    "crop_overflow_right",
    "crop_visible_fraction",
    "crop_has_padding",
    "mask_touching_square_crop_border",
    "mask_touching_rect_border",
    "mask_touching_image_border",
)

MULTISCALE_RECT_IMAGE_MASK_RANKER_FEATURE_COLUMNS = (
    "tight_mask_area_fraction",
    "tight_mask_extent",
    "tight_mask_perimeter_area_ratio",
    "tight_mask_border_touch_fraction",
    "tight_mask_centroid_offset_norm",
    "tight_raw_contrast",
    "tight_raw_contrast_norm",
    "tight_raw_boundary_contrast",
    "far_mask_area_fraction",
    "far_mask_extent",
    "far_mask_perimeter_area_ratio",
    "far_mask_border_touch_fraction",
    "far_mask_centroid_offset_norm",
    "far_raw_contrast",
    "far_raw_contrast_norm",
    "far_raw_boundary_contrast",
)

FULL_IMAGE_MASK_RANKER_FEATURE_COLUMNS = (
    "full_mask_area_pixels",
    "full_mask_area_fraction",
    "full_mask_retained_fraction",
    "full_mask_bbox_aspect_ratio",
    "full_mask_bbox_area",
    "full_mask_extent",
    "full_mask_perimeter_area_ratio",
    "full_mask_component_count",
    "full_mask_border_touch_fraction",
    "full_mask_centroid_offset_norm",
    "full_raw_contrast",
    "full_raw_contrast_norm",
    "full_raw_boundary_contrast",
    "ranker_cluster_full_raw_contrast_norm_rank",
    "ranker_cluster_full_boundary_contrast_rank",
    "ranker_cluster_full_mask_extent_rank",
    "full_mask_area_log_ratio_to_cluster_median",
    "full_mask_area_abs_log_ratio_to_cluster_median",
)

FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS = (
    "full_raw_contrast_to_noise",
    "full_raw_boundary_gradient_mean",
    "full_raw_boundary_gradient_contrast",
    "full_mask_hole_fraction",
    "full_mask_eccentricity",
)


@dataclass(frozen=True)
class CandidateRankerConfig:
    target_column: str = "true_iou"
    train_splits: tuple[str, ...] = ("train",)
    calibration_splits: tuple[str, ...] = ("validation",)
    feature_columns: tuple[str, ...] = DEFAULT_RANKER_FEATURE_COLUMNS
    categorical_columns: tuple[str, ...] = ()
    min_training_rows: int = 3
    min_calibration_rows: int = 3
    model_type: str = "ridge"
    ridge_alpha: float = 1.0
    mlp_hidden_layer_sizes: tuple[int, ...] = (64, 32)
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 500
    random_state: int = 13
    cluster_distance: float = 4.0
    use_source_cluster_quality_prior: bool = False


def train_candidate_ranker(
    features: pd.DataFrame,
    *,
    output_model: Path | None = None,
    config: CandidateRankerConfig = CandidateRankerConfig(),
) -> dict[str, Any]:
    features = add_cluster_ranker_features(
        features, cluster_distance=config.cluster_distance
    )
    configured_feature_columns = list(config.feature_columns)
    source_cluster_quality_priors: dict[str, float] = {}
    dataset_cluster_quality_prior = 0.5
    if config.use_source_cluster_quality_prior:
        features, source_cluster_quality_priors, dataset_cluster_quality_prior = (
            add_source_cluster_quality_prior(
                features,
                train_splits=config.train_splits,
                target_column=config.target_column,
            )
        )
        configured_feature_columns.extend(
            ["source_cluster_quality_prior", "ranker_cluster_source_quality_rank"]
        )
    feature_columns = _available_feature_columns(
        features, tuple(configured_feature_columns)
    )
    if not feature_columns:
        raise ValueError("No configured ranker feature columns are present.")
    categorical_columns = _available_feature_columns(
        features, config.categorical_columns
    )
    if config.target_column not in features.columns:
        raise ValueError(f"Missing target column: {config.target_column}")

    target = pd.to_numeric(features[config.target_column], errors="coerce")
    train_mask = _split_mask(features, config.train_splits) & target.notna()
    if int(train_mask.sum()) < config.min_training_rows:
        raise ValueError(
            f"Need at least {config.min_training_rows} training rows, "
            f"found {int(train_mask.sum())}."
        )

    model_columns = feature_columns + categorical_columns
    model = _build_model(
        numeric_columns=feature_columns,
        categorical_columns=categorical_columns,
        config=config,
    )
    model.fit(features.loc[train_mask, model_columns], target.loc[train_mask])

    calibrator = _fit_isotonic_calibrator(
        model=model,
        features=features,
        target=target,
        model_columns=model_columns,
        config=config,
    )
    payload = {
        "model": model,
        "calibrator": calibrator,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "model_columns": model_columns,
        "config": asdict(config),
        "source_cluster_quality_priors": source_cluster_quality_priors,
        "dataset_cluster_quality_prior": dataset_cluster_quality_prior,
        "train_rows": int(train_mask.sum()),
        "calibration_rows": int(
            (_split_mask(features, config.calibration_splits) & target.notna()).sum()
        ),
    }

    if output_model is not None:
        output_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, output_model)
    return payload


def score_with_candidate_ranker(
    features: pd.DataFrame,
    *,
    model_payload: dict[str, Any] | Path,
) -> pd.DataFrame:
    payload = _load_payload(model_payload)
    config = payload.get("config", {})
    features = add_cluster_ranker_features(
        features, cluster_distance=float(config.get("cluster_distance", 4.0))
    )
    if bool(config.get("use_source_cluster_quality_prior", False)):
        features = apply_source_cluster_quality_prior(
            features,
            source_cluster_quality_priors=payload.get(
                "source_cluster_quality_priors", {}
            ),
            dataset_cluster_quality_prior=float(
                payload.get("dataset_cluster_quality_prior", 0.5)
            ),
        )
    model_columns = list(payload.get("model_columns", payload["feature_columns"]))
    missing = sorted(set(model_columns).difference(features.columns))
    if missing:
        raise ValueError(f"Feature table is missing ranker columns: {missing}")

    raw_score = payload["model"].predict(features[model_columns])
    calibrator = payload.get("calibrator")
    if calibrator is not None:
        score = calibrator.predict(raw_score)
    else:
        score = raw_score

    scored = features.copy()
    scored["candidate_ranker_raw_score"] = raw_score
    scored["candidate_ranker_score"] = np.clip(score, 0.0, 1.0)
    if "image_id" in scored.columns:
        scored["candidate_ranker_rank_in_image"] = scored.groupby(
            "image_id", dropna=False
        )["candidate_ranker_score"].rank(method="average", pct=True)
    else:
        scored["candidate_ranker_rank_in_image"] = scored[
            "candidate_ranker_score"
        ].rank(method="average", pct=True)
    if "ranker_cluster_id" in scored.columns:
        scored["candidate_ranker_rank_in_cluster"] = scored.groupby(
            "ranker_cluster_id", dropna=False
        )["candidate_ranker_score"].rank(method="average", pct=True)
    else:
        scored["candidate_ranker_rank_in_cluster"] = scored[
            "candidate_ranker_rank_in_image"
        ]
    return scored


def add_cluster_ranker_features(
    features: pd.DataFrame, *, cluster_distance: float = 4.0
) -> pd.DataFrame:
    if not {"image_id", "center_y", "center_x"}.issubset(features.columns):
        return features.copy()

    augmented = features.copy()
    augmented["ranker_cluster_id"] = _assign_center_clusters(
        augmented, max_distance=cluster_distance
    )
    augmented["ranker_cluster_size"] = augmented.groupby(
        "ranker_cluster_id", dropna=False
    )["ranker_cluster_id"].transform("size")
    augmented["ranker_cluster_qa_rank"] = _rank_within_cluster(augmented, "qa_score")
    augmented["ranker_cluster_source_prior_rank"] = _rank_within_cluster(
        augmented, "source_train_mean_iou"
    )
    source_temporal_distance = _numeric_series(
        augmented, "source_temporal_min_center_dist_norm"
    )
    any_temporal_distance = _numeric_series(augmented, "temporal_min_center_dist_norm")
    temporal_distance = source_temporal_distance.fillna(any_temporal_distance)
    source_temporal_support = _numeric_series(
        augmented, "source_temporal_support_score"
    )
    any_temporal_support = _numeric_series(augmented, "temporal_support_score")
    temporal_support = source_temporal_support.fillna(any_temporal_support)
    augmented["_ranker_temporal_score"] = (
        1.0 / (1.0 + temporal_distance)
    ).clip(0.0, 1.0)
    augmented["_ranker_temporal_support_score"] = temporal_support.fillna(
        augmented["_ranker_temporal_score"]
    ).clip(0.0, 1.0)
    augmented["ranker_cluster_temporal_rank"] = augmented.groupby(
        "ranker_cluster_id", dropna=False
    )["_ranker_temporal_score"].rank(method="average", pct=True)
    augmented["ranker_cluster_temporal_support_rank"] = augmented.groupby(
        "ranker_cluster_id", dropna=False
    )["_ranker_temporal_support_score"].rank(method="average", pct=True)
    _add_image_cluster_relative_features(augmented)
    return augmented.drop(
        columns=["_ranker_temporal_score", "_ranker_temporal_support_score"]
    )


def _add_image_cluster_relative_features(features: pd.DataFrame) -> None:
    if "ranker_cluster_id" not in features.columns:
        return

    _add_prefixed_cluster_relative_features(features, prefix="rect")
    _add_prefixed_cluster_relative_features(features, prefix="full")


def _add_prefixed_cluster_relative_features(
    features: pd.DataFrame, *, prefix: str
) -> None:
    contrast_column = f"{prefix}_raw_contrast_norm"
    boundary_column = f"{prefix}_raw_boundary_contrast"
    extent_column = f"{prefix}_mask_extent"
    area_column = f"{prefix}_mask_area_pixels"

    if contrast_column in features.columns:
        features[f"ranker_cluster_{prefix}_raw_contrast_norm_rank"] = (
            _rank_within_cluster(features, contrast_column)
        )
    if boundary_column in features.columns:
        features[f"ranker_cluster_{prefix}_boundary_contrast_rank"] = (
            _rank_within_cluster(features, boundary_column)
        )
    if extent_column in features.columns:
        features[f"ranker_cluster_{prefix}_mask_extent_rank"] = _rank_within_cluster(
            features, extent_column
        )
    if area_column in features.columns:
        area = _numeric_series(features, area_column)
        median_area = area.groupby(features["ranker_cluster_id"], dropna=False).transform(
            "median"
        )
        ratio = area / median_area.replace(0, np.nan)
        log_ratio = np.log(ratio.replace(0, np.nan))
        features[f"{prefix}_mask_area_log_ratio_to_cluster_median"] = log_ratio
        features[f"{prefix}_mask_area_abs_log_ratio_to_cluster_median"] = np.abs(
            log_ratio
        )


def add_source_cluster_quality_prior(
    features: pd.DataFrame,
    *,
    train_splits: tuple[str, ...],
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, float], float]:
    if not {"ranker_cluster_id", "candidate_source", target_column}.issubset(
        features.columns
    ):
        augmented = features.copy()
        augmented["source_cluster_quality_prior"] = 0.5
        augmented["ranker_cluster_source_quality_rank"] = _rank_within_cluster(
            augmented, "source_cluster_quality_prior"
        )
        return augmented, {}, 0.5

    target = pd.to_numeric(features[target_column], errors="coerce")
    train_mask = _split_mask(features, train_splits) & target.notna()
    train_rows = features.loc[train_mask].copy()
    if train_rows.empty:
        augmented = apply_source_cluster_quality_prior(
            features,
            source_cluster_quality_priors={},
            dataset_cluster_quality_prior=0.5,
        )
        return augmented, {}, 0.5

    train_target = pd.to_numeric(train_rows[target_column], errors="coerce")
    train_rows["_cluster_oracle_iou"] = train_target.groupby(
        train_rows["ranker_cluster_id"], dropna=False
    ).transform("max")
    train_rows["_cluster_regret"] = (
        train_rows["_cluster_oracle_iou"] - train_target
    ).clip(lower=0.0, upper=1.0)
    source_regret = train_rows.groupby("candidate_source")["_cluster_regret"].mean()
    source_quality = (1.0 - source_regret).clip(lower=0.0, upper=1.0)
    dataset_quality = float(
        np.clip(1.0 - float(train_rows["_cluster_regret"].mean()), 0.0, 1.0)
    )
    priors = {str(source): float(value) for source, value in source_quality.items()}
    augmented = apply_source_cluster_quality_prior(
        features,
        source_cluster_quality_priors=priors,
        dataset_cluster_quality_prior=dataset_quality,
    )
    return augmented, priors, dataset_quality


def apply_source_cluster_quality_prior(
    features: pd.DataFrame,
    *,
    source_cluster_quality_priors: dict[str, float],
    dataset_cluster_quality_prior: float,
) -> pd.DataFrame:
    augmented = features.copy()
    if "candidate_source" not in augmented.columns:
        augmented["source_cluster_quality_prior"] = dataset_cluster_quality_prior
    else:
        augmented["source_cluster_quality_prior"] = (
            augmented["candidate_source"]
            .astype(str)
            .map(source_cluster_quality_priors)
            .fillna(dataset_cluster_quality_prior)
        )
    augmented["source_cluster_quality_prior"] = pd.to_numeric(
        augmented["source_cluster_quality_prior"], errors="coerce"
    ).fillna(0.5)
    if "ranker_cluster_id" in augmented.columns:
        augmented["ranker_cluster_source_quality_rank"] = _rank_within_cluster(
            augmented, "source_cluster_quality_prior"
        )
    else:
        augmented["ranker_cluster_source_quality_rank"] = augmented[
            "source_cluster_quality_prior"
        ].rank(method="average", pct=True)
    return augmented


def write_ranker_scored_features(
    features_path: Path,
    model_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    features = _read_table(features_path)
    scored = score_with_candidate_ranker(features, model_payload=model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        scored.to_parquet(output_path, index=False)
    else:
        scored.to_csv(output_path, index=False)
    return scored


def _fit_isotonic_calibrator(
    *,
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    model_columns: list[str],
    config: CandidateRankerConfig,
) -> IsotonicRegression | None:
    calibration_mask = _split_mask(features, config.calibration_splits) & target.notna()
    if int(calibration_mask.sum()) < config.min_calibration_rows:
        return None

    raw = model.predict(features.loc[calibration_mask, model_columns])
    y = target.loc[calibration_mask].clip(lower=0.0, upper=1.0)
    if pd.Series(raw).nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return None

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw, y)
    return calibrator


def _available_feature_columns(
    features: pd.DataFrame, configured_columns: tuple[str, ...]
) -> list[str]:
    return [column for column in configured_columns if column in features.columns]


def _build_model(
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: CandidateRankerConfig,
) -> Any:
    estimator = _build_estimator(config)
    if not categorical_columns:
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            estimator,
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_columns,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )
    return make_pipeline(preprocessor, estimator)


def _build_estimator(config: CandidateRankerConfig) -> Any:
    if config.model_type == "ridge":
        return Ridge(alpha=config.ridge_alpha)
    if config.model_type == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=config.mlp_hidden_layer_sizes,
            alpha=config.mlp_alpha,
            max_iter=config.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=20,
            random_state=config.random_state,
        )
    raise ValueError(f"Unsupported candidate ranker model_type: {config.model_type!r}")


def _split_mask(features: pd.DataFrame, splits: tuple[str, ...]) -> pd.Series:
    if "split" not in features.columns:
        return pd.Series(True, index=features.index)
    return features["split"].astype(str).isin(splits)


def _rank_within_cluster(features: pd.DataFrame, column: str) -> pd.Series:
    if column not in features.columns:
        return pd.Series(np.nan, index=features.index, dtype=float)
    return features.groupby("ranker_cluster_id", dropna=False)[column].rank(
        method="average", pct=True
    )


def _numeric_series(features: pd.DataFrame, column: str) -> pd.Series:
    if column not in features.columns:
        return pd.Series(np.nan, index=features.index, dtype=float)
    return pd.to_numeric(features[column], errors="coerce")


def _assign_center_clusters(
    features: pd.DataFrame, *, max_distance: float
) -> pd.Series:
    cluster_ids = pd.Series(index=features.index, dtype=object)
    for image_id, group in features.groupby("image_id", dropna=False, sort=False):
        centers = group[["center_y", "center_x"]].to_numpy(dtype=float)
        labels = _connected_components(centers, max_distance=max_distance)
        cluster_ids.loc[group.index] = [
            f"{image_id}__cluster_{label:04d}" for label in labels
        ]
    return cluster_ids


def _connected_components(points: np.ndarray, *, max_distance: float) -> list[int]:
    n_points = len(points)
    parent = list(range(n_points))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(n_points):
        if np.isnan(points[left]).any():
            continue
        for right in range(left + 1, n_points):
            if np.isnan(points[right]).any():
                continue
            if float(np.linalg.norm(points[left] - points[right])) <= max_distance:
                union(left, right)

    roots: dict[int, int] = {}
    labels = []
    for index in range(n_points):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels


def _load_payload(model_payload: dict[str, Any] | Path) -> dict[str, Any]:
    if isinstance(model_payload, (str, Path)):
        return joblib.load(model_payload)
    return model_payload


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or apply a calibrated candidate ranker."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("features", type=Path)
    train_parser.add_argument("--output-model", required=True, type=Path)
    train_parser.add_argument("--train-split", action="append", default=None)
    train_parser.add_argument("--calibration-split", action="append", default=None)
    train_parser.add_argument("--target-column", default="true_iou")
    train_parser.add_argument(
        "--model-type",
        choices=("ridge", "mlp"),
        default="ridge",
        help="Regressor used before isotonic calibration.",
    )
    train_parser.add_argument("--mlp-hidden-layer-size", action="append", type=int)
    train_parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    train_parser.add_argument("--mlp-max-iter", type=int, default=500)
    train_parser.add_argument(
        "--use-source-identity",
        action="store_true",
        help="Include canonical candidate_source as a categorical ranker feature.",
    )
    train_parser.add_argument(
        "--use-source-cluster-quality-prior",
        action="store_true",
        help="Include source-level local-cluster regret quality prior.",
    )

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("features", type=Path)
    score_parser.add_argument("--model", required=True, type=Path)
    score_parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "train":
        features = _read_table(args.features)
        config = CandidateRankerConfig(
            target_column=args.target_column,
            train_splits=tuple(args.train_split or ("train",)),
            calibration_splits=tuple(args.calibration_split or ("validation",)),
            model_type=args.model_type,
            mlp_hidden_layer_sizes=tuple(args.mlp_hidden_layer_size or (64, 32)),
            mlp_alpha=args.mlp_alpha,
            mlp_max_iter=args.mlp_max_iter,
            categorical_columns=("candidate_source",)
            if args.use_source_identity
            else (),
            use_source_cluster_quality_prior=args.use_source_cluster_quality_prior,
        )
        payload = train_candidate_ranker(
            features, output_model=args.output_model, config=config
        )
        print(
            "Wrote candidate ranker to "
            f"{args.output_model} with {payload['train_rows']} train rows and "
            f"{payload['calibration_rows']} calibration rows"
        )
    elif args.command == "score":
        scored = write_ranker_scored_features(args.features, args.model, args.output)
        print(f"Wrote {len(scored)} ranker-scored rows to {args.output}")


if __name__ == "__main__":
    main()
