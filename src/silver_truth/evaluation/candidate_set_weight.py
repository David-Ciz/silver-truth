from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from silver_truth.evaluation.candidate_ranker import (
    CROP_GEOMETRY_RANKER_FEATURE_COLUMNS,
    DEFAULT_RANKER_FEATURE_COLUMNS,
    FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS,
    FULL_IMAGE_MASK_RANKER_FEATURE_COLUMNS,
    RECT_IMAGE_MASK_RANKER_FEATURE_COLUMNS,
    TEMPORAL_SUPPORT_RANKER_FEATURE_COLUMNS,
    add_cluster_ranker_features,
)


DEFAULT_CANDIDATE_SET_WEIGHT_FEATURE_COLUMNS = (
    "pairwise_cluster_score",
    "pairwise_cluster_rank",
    "local_image_support_score",
    *DEFAULT_RANKER_FEATURE_COLUMNS,
    *TEMPORAL_SUPPORT_RANKER_FEATURE_COLUMNS,
    *RECT_IMAGE_MASK_RANKER_FEATURE_COLUMNS,
    *FULL_IMAGE_MASK_RANKER_FEATURE_COLUMNS,
    *FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS,
    *CROP_GEOMETRY_RANKER_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class CandidateSetWeightConfig:
    target_column: str = "true_iou"
    train_splits: tuple[str, ...] = ("train", "validation")
    feature_columns: tuple[str, ...] = DEFAULT_CANDIDATE_SET_WEIGHT_FEATURE_COLUMNS
    categorical_columns: tuple[str, ...] = ("dataset", "candidate_source")
    min_training_rows: int = 10
    model_type: str = "mlp"
    ridge_alpha: float = 1.0
    mlp_hidden_layer_sizes: tuple[int, ...] = (64, 32)
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 500
    random_state: int = 13
    set_context_stats: tuple[str, ...] = ("mean", "std", "min", "max")
    categorical_dropout_columns: tuple[str, ...] = ()
    categorical_dropout_probability: float = 0.0
    categorical_dropout_value: str = "__candidate_set_masked__"
    cluster_distance: float = 4.0
    target_temperature: float = 0.08
    min_score: float = 1e-3
    deep_sets_hidden_dim: int = 64
    deep_sets_epochs: int = 120
    deep_sets_learning_rate: float = 1e-3
    deep_sets_weight_decay: float = 1e-4
    deep_sets_batch_clusters: int = 32
    deep_sets_device: str = "auto"
    deep_sets_bad_mass_penalty_weight: float = 0.0
    deep_sets_bad_mass_margin: float = 0.10
    deep_sets_oversize_bad_mass_penalty_weight: float = 0.0
    deep_sets_oversize_bad_mass_margin: float = 0.10
    deep_sets_oversize_area_column: str = "full_mask_area_log_ratio_to_cluster_median"
    deep_sets_oversize_area_log_ratio_threshold: float = 0.25


def train_candidate_set_weight_predictor(
    features: pd.DataFrame,
    *,
    output_model: Path | None = None,
    config: CandidateSetWeightConfig = CandidateSetWeightConfig(),
) -> dict[str, Any]:
    augmented = add_candidate_set_relative_features(features, config=config)
    if _uses_set_context(config):
        augmented = add_candidate_set_context_features(augmented, config=config)
    feature_columns = _available_columns(augmented, config.feature_columns)
    relative_columns = [
        column
        for column in augmented.columns
        if column.startswith("candidate_set_rank__")
        or column.startswith("candidate_set_centered__")
        or column.startswith("candidate_set_gap_to_best__")
    ]
    context_columns = [
        column
        for column in augmented.columns
        if column.startswith("candidate_set_context__")
    ]
    categorical_columns = _available_columns(augmented, config.categorical_columns)
    if config.target_column not in augmented.columns:
        raise ValueError(f"Missing target column: {config.target_column}")

    target = _candidate_set_target(
        augmented,
        target_column=config.target_column,
        temperature=config.target_temperature,
    )
    train_mask = _split_mask(augmented, config.train_splits) & target.notna()
    if int(train_mask.sum()) < config.min_training_rows:
        raise ValueError(
            f"Need at least {config.min_training_rows} training rows, "
            f"found {int(train_mask.sum())}."
        )

    model_feature_columns = [
        column
        for column in feature_columns + relative_columns + context_columns
        if _numeric_feature_series(augmented.loc[train_mask], column)
        .notna()
        .any()
    ]
    if not model_feature_columns:
        raise ValueError("No candidate-set predictor feature columns are present.")

    model_columns = model_feature_columns + categorical_columns
    if config.model_type == "deep_sets":
        payload = _train_deep_sets_predictor(
            augmented=augmented,
            target=target,
            train_mask=train_mask,
            numeric_columns=model_feature_columns,
            categorical_columns=categorical_columns,
            model_columns=model_columns,
            config=config,
        )
        if output_model is not None:
            output_model.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, output_model)
        return payload

    model = _build_model(
        numeric_columns=model_feature_columns,
        categorical_columns=categorical_columns,
        config=config,
    )
    train_features = augmented.loc[train_mask, model_columns].copy()
    train_features = _apply_categorical_dropout(train_features, config=config)
    model.fit(train_features, target.loc[train_mask])
    payload = {
        "model": model,
        "feature_columns": model_feature_columns,
        "categorical_columns": categorical_columns,
        "model_columns": model_columns,
        "config": asdict(config),
        "train_rows": int(train_mask.sum()),
        "target_mean": float(target.loc[train_mask].mean()),
    }
    if output_model is not None:
        output_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, output_model)
    return payload


def score_with_candidate_set_weight_predictor(
    features: pd.DataFrame,
    *,
    model_payload: dict[str, Any] | Path,
) -> pd.DataFrame:
    payload = _load_payload(model_payload)
    config = CandidateSetWeightConfig(**payload.get("config", {}))
    augmented = add_candidate_set_relative_features(features, config=config)
    if _uses_set_context(config):
        augmented = add_candidate_set_context_features(augmented, config=config)
    model_columns = list(payload.get("model_columns", payload["feature_columns"]))
    missing = sorted(set(model_columns).difference(augmented.columns))
    if missing:
        raise ValueError(f"Feature table is missing candidate-set columns: {missing}")

    if payload.get("model_type") == "deep_sets":
        raw_score = _predict_deep_sets(augmented, payload=payload)
    else:
        raw_score = payload["model"].predict(augmented[model_columns])
    scored = augmented.copy()
    scored["candidate_set_weight_raw_score"] = raw_score
    scored["candidate_set_weight_score"] = np.clip(
        raw_score, config.min_score, 1.0
    )
    scored["candidate_set_weight_rank"] = scored.groupby(
        "ranker_cluster_id", dropna=False
    )["candidate_set_weight_score"].rank(method="average", pct=True)
    scored["candidate_set_weight_sum"] = scored.groupby(
        "ranker_cluster_id", dropna=False
    )["candidate_set_weight_score"].transform("sum")
    scored["candidate_set_weight_normalized"] = (
        scored["candidate_set_weight_score"]
        / scored["candidate_set_weight_sum"].replace(0, np.nan)
    ).fillna(1.0)
    return scored


def add_candidate_set_relative_features(
    features: pd.DataFrame,
    *,
    config: CandidateSetWeightConfig = CandidateSetWeightConfig(),
) -> pd.DataFrame:
    augmented = add_cluster_ranker_features(
        features, cluster_distance=config.cluster_distance
    )
    if "ranker_cluster_id" not in augmented.columns:
        augmented["ranker_cluster_id"] = "all_candidates"

    relative: dict[str, pd.Series] = {}
    for column in _available_columns(augmented, config.feature_columns):
        values = pd.to_numeric(augmented[column], errors="coerce")
        grouped = values.groupby(augmented["ranker_cluster_id"], dropna=False)
        cluster_mean = grouped.transform("mean")
        cluster_best = grouped.transform("max")
        relative[f"candidate_set_rank__{column}"] = grouped.rank(
            method="average", pct=True
        )
        relative[f"candidate_set_centered__{column}"] = values - cluster_mean
        relative[f"candidate_set_gap_to_best__{column}"] = cluster_best - values
    if not relative:
        return augmented.copy()
    return pd.concat([augmented, pd.DataFrame(relative, index=augmented.index)], axis=1)


def add_candidate_set_context_features(
    features: pd.DataFrame,
    *,
    config: CandidateSetWeightConfig = CandidateSetWeightConfig(),
) -> pd.DataFrame:
    """Add permutation-invariant cluster summaries for set-context predictors."""
    augmented = features.copy()
    if "ranker_cluster_id" not in augmented.columns:
        augmented = add_cluster_ranker_features(
            augmented, cluster_distance=config.cluster_distance
        )
    if "ranker_cluster_id" not in augmented.columns:
        augmented["ranker_cluster_id"] = "all_candidates"

    context: dict[str, pd.Series] = {}
    cluster_ids = augmented["ranker_cluster_id"]
    context["candidate_set_context__cluster_size"] = cluster_ids.groupby(
        cluster_ids, dropna=False
    ).transform("size")

    stats = set(config.set_context_stats)
    for column in _available_columns(augmented, config.feature_columns):
        values = pd.to_numeric(augmented[column], errors="coerce")
        grouped = values.groupby(cluster_ids, dropna=False)
        if "mean" in stats:
            context[f"candidate_set_context__mean__{column}"] = grouped.transform(
                "mean"
            )
        if "std" in stats:
            context[f"candidate_set_context__std__{column}"] = (
                grouped.transform("std").fillna(0.0)
            )
        if "min" in stats:
            context[f"candidate_set_context__min__{column}"] = grouped.transform("min")
        if "max" in stats:
            context[f"candidate_set_context__max__{column}"] = grouped.transform("max")

    return pd.concat([augmented, pd.DataFrame(context, index=augmented.index)], axis=1)


def write_candidate_set_weighted_features(
    feature_paths: list[Path],
    *,
    output_root: Path,
    relative_root: Path | None = None,
    output_model: Path | None = None,
    config: CandidateSetWeightConfig = CandidateSetWeightConfig(),
) -> pd.DataFrame:
    if not feature_paths:
        raise ValueError("No feature paths were provided.")

    frames = []
    for path in feature_paths:
        frame = pd.read_parquet(path).copy()
        frame["_candidate_set_source_path"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    payload = train_candidate_set_weight_predictor(
        combined, output_model=output_model, config=config
    )

    output_rows = []
    for path, frame in zip(feature_paths, frames, strict=True):
        scored = score_with_candidate_set_weight_predictor(
            frame.drop(columns=["_candidate_set_source_path"]),
            model_payload=payload,
        )
        output_path = _output_path_for(path, output_root, relative_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(output_path, index=False)
        output_rows.append(
            {
                "feature_path": str(path),
                "output_path": str(output_path),
                "rows": int(len(scored)),
                "train_rows": int(payload["train_rows"]),
                "mean_candidate_set_weight_score": float(
                    scored["candidate_set_weight_score"].mean()
                ),
            }
        )
    return pd.DataFrame.from_records(output_rows)


def write_leave_one_dataset_out_weighted_features(
    feature_paths: list[Path],
    *,
    output_root: Path,
    relative_root: Path | None = None,
    output_model_root: Path | None = None,
    config: CandidateSetWeightConfig = CandidateSetWeightConfig(),
) -> pd.DataFrame:
    if not feature_paths:
        raise ValueError("No feature paths were provided.")

    frames = []
    for path in feature_paths:
        frame = pd.read_parquet(path).copy()
        if "dataset" not in frame.columns:
            raise ValueError(f"{path} is missing required column 'dataset'.")
        frame["_candidate_set_source_path"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    datasets = sorted(str(value) for value in combined["dataset"].dropna().unique())
    if len(datasets) < 2:
        raise ValueError("Leave-one-dataset-out scoring needs at least two datasets.")

    payloads: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        train_source = combined[~combined["dataset"].astype(str).eq(dataset)].copy()
        model_path = (
            output_model_root / f"{dataset}_candidate_set_weight_predictor.joblib"
            if output_model_root is not None
            else None
        )
        payloads[dataset] = train_candidate_set_weight_predictor(
            train_source.drop(columns=["_candidate_set_source_path"]),
            output_model=model_path,
            config=config,
        )

    output_rows = []
    for path, frame in zip(feature_paths, frames, strict=True):
        datasets_in_frame = frame["dataset"].astype(str).dropna().unique()
        if len(datasets_in_frame) != 1:
            raise ValueError(f"{path} must contain exactly one dataset for LODO scoring.")
        dataset = str(datasets_in_frame[0])
        scored = score_with_candidate_set_weight_predictor(
            frame.drop(columns=["_candidate_set_source_path"]),
            model_payload=payloads[dataset],
        )
        output_path = _output_path_for(path, output_root, relative_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(output_path, index=False)
        output_rows.append(
            {
                "feature_path": str(path),
                "output_path": str(output_path),
                "held_out_dataset": dataset,
                "rows": int(len(scored)),
                "train_rows": int(payloads[dataset]["train_rows"]),
                "mean_candidate_set_weight_score": float(
                    scored["candidate_set_weight_score"].mean()
                ),
            }
        )
    return pd.DataFrame.from_records(output_rows)


def write_leave_target_video_out_weighted_features(
    feature_paths: list[Path],
    *,
    output_root: Path,
    relative_root: Path | None = None,
    output_model_root: Path | None = None,
    config: CandidateSetWeightConfig = CandidateSetWeightConfig(),
) -> pd.DataFrame:
    if not feature_paths:
        raise ValueError("No feature paths were provided.")

    frames = []
    for path in feature_paths:
        frame = pd.read_parquet(path).copy()
        missing = {"dataset", "fold", "campaign_number", "split"}.difference(
            frame.columns
        )
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frame["_candidate_set_source_path"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    output_rows = []
    for path, frame in zip(feature_paths, frames, strict=True):
        metadata = _target_video_metadata(frame, path)
        target_dataset = metadata["dataset"]
        target_campaign = metadata["test_campaign"]
        target_fold = metadata["fold"]
        train_source = combined[
            ~(
                combined["dataset"].astype(str).eq(target_dataset)
                & combined["campaign_number"].astype(str).eq(target_campaign)
            )
        ].copy()
        model_path = (
            output_model_root
            / f"{target_dataset}_{target_fold}_leave_target_video_out.joblib"
            if output_model_root is not None
            else None
        )
        payload = train_candidate_set_weight_predictor(
            train_source.drop(columns=["_candidate_set_source_path"]),
            output_model=model_path,
            config=config,
        )
        scored = score_with_candidate_set_weight_predictor(
            frame.drop(columns=["_candidate_set_source_path"]),
            model_payload=payload,
        )
        output_path = _output_path_for(path, output_root, relative_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(output_path, index=False)
        output_rows.append(
            {
                "feature_path": str(path),
                "output_path": str(output_path),
                "held_out_dataset": target_dataset,
                "held_out_fold": target_fold,
                "held_out_campaign": target_campaign,
                "rows": int(len(scored)),
                "train_rows": int(payload["train_rows"]),
                "mean_candidate_set_weight_score": float(
                    scored["candidate_set_weight_score"].mean()
                ),
            }
        )
    return pd.DataFrame.from_records(output_rows)


def _candidate_set_target(
    features: pd.DataFrame,
    *,
    target_column: str,
    temperature: float,
) -> pd.Series:
    target = pd.to_numeric(features[target_column], errors="coerce")
    best = target.groupby(features["ranker_cluster_id"], dropna=False).transform("max")
    temp = max(float(temperature), 1e-6)
    return np.exp(((target - best) / temp).clip(lower=-50.0, upper=0.0))


def _target_video_metadata(frame: pd.DataFrame, path: Path) -> dict[str, str]:
    datasets = frame["dataset"].astype(str).dropna().unique()
    folds = frame["fold"].astype(str).dropna().unique()
    if len(datasets) != 1 or len(folds) != 1:
        raise ValueError(f"{path} must contain exactly one dataset and one fold.")
    test = frame[frame["split"].astype(str).eq("test")]
    campaigns = test["campaign_number"].astype(str).dropna().unique()
    if len(campaigns) != 1:
        raise ValueError(
            f"{path} must contain exactly one held-out test campaign, "
            f"found {sorted(campaigns)}."
        )
    return {
        "dataset": str(datasets[0]),
        "fold": str(folds[0]),
        "test_campaign": str(campaigns[0]),
    }


def _build_model(
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: CandidateSetWeightConfig,
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


def _build_estimator(config: CandidateSetWeightConfig) -> Any:
    if config.model_type == "ridge":
        return Ridge(alpha=config.ridge_alpha)
    if config.model_type in {"mlp", "set_mlp"}:
        return MLPRegressor(
            hidden_layer_sizes=config.mlp_hidden_layer_sizes,
            alpha=config.mlp_alpha,
            max_iter=config.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=20,
            random_state=config.random_state,
        )
    if config.model_type == "deep_sets":
        raise AssertionError("deep_sets is built by _train_deep_sets_predictor.")
    raise ValueError(f"Unsupported candidate-set model_type: {config.model_type!r}")


def _uses_set_context(config: CandidateSetWeightConfig) -> bool:
    return config.model_type == "set_mlp"


def _split_mask(features: pd.DataFrame, splits: tuple[str, ...]) -> pd.Series:
    if "split" not in features.columns:
        return pd.Series(True, index=features.index)
    return features["split"].astype(str).isin(splits)


def _available_columns(features: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    seen = set()
    available = []
    for column in columns:
        if column in features.columns and column not in seen:
            available.append(column)
            seen.add(column)
    return available


def _numeric_feature_series(features: pd.DataFrame, column: str) -> pd.Series:
    values = features.loc[:, column]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce")


def _apply_categorical_dropout(
    features: pd.DataFrame,
    *,
    config: CandidateSetWeightConfig,
) -> pd.DataFrame:
    probability = float(config.categorical_dropout_probability)
    if probability <= 0.0 or not config.categorical_dropout_columns:
        return features
    if probability > 1.0:
        raise ValueError("categorical_dropout_probability must be between 0 and 1.")

    dropout_columns = _available_columns(features, config.categorical_dropout_columns)
    if not dropout_columns:
        return features

    rng = np.random.default_rng(config.random_state)
    dropped = features.copy()
    for column in dropout_columns:
        mask = rng.random(len(dropped)) < probability
        dropped.loc[mask, column] = config.categorical_dropout_value
    return dropped


class _DeepSetsWeightRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.phi(x)
        context = encoded.mean(dim=0, keepdim=True).expand_as(encoded)
        return self.rho(torch.cat([encoded, context], dim=1)).squeeze(1)


def _train_deep_sets_predictor(
    *,
    augmented: pd.DataFrame,
    target: pd.Series,
    train_mask: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_columns: list[str],
    config: CandidateSetWeightConfig,
) -> dict[str, Any]:
    train_features = augmented.loc[train_mask, model_columns].copy()
    train_features = _apply_categorical_dropout(train_features, config=config)
    preprocess = _fit_deep_sets_preprocess(
        train_features,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    train_matrix = _deep_sets_matrix(
        train_features,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        preprocess=preprocess,
    )
    train_target = target.loc[train_mask].astype(float).to_numpy()
    train_true_iou = (
        pd.to_numeric(augmented.loc[train_mask, config.target_column], errors="coerce")
        .astype(float)
        .to_numpy()
    )
    train_oversize_area = _deep_sets_oversize_area_values(
        augmented.loc[train_mask], config=config
    )
    cluster_ids = augmented.loc[train_mask, "ranker_cluster_id"].astype(str).to_numpy()
    clusters = _deep_sets_clusters(
        train_matrix,
        train_target,
        train_true_iou,
        train_oversize_area,
        cluster_ids,
        bad_mass_margin=config.deep_sets_bad_mass_margin,
        oversize_bad_mass_margin=config.deep_sets_oversize_bad_mass_margin,
        oversize_area_log_ratio_threshold=(
            config.deep_sets_oversize_area_log_ratio_threshold
        ),
    )
    if not clusters:
        raise ValueError("No non-empty candidate clusters available for deep_sets.")

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    device = _resolve_torch_device(config.deep_sets_device)
    model = _DeepSetsWeightRegressor(
        input_dim=train_matrix.shape[1],
        hidden_dim=config.deep_sets_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.deep_sets_learning_rate,
        weight_decay=config.deep_sets_weight_decay,
    )
    loss_fn = nn.MSELoss()
    bad_mass_weight = float(config.deep_sets_bad_mass_penalty_weight)
    oversize_bad_mass_weight = float(
        config.deep_sets_oversize_bad_mass_penalty_weight
    )
    rng = np.random.default_rng(config.random_state)
    losses: list[float] = []

    for _ in range(config.deep_sets_epochs):
        order = rng.permutation(len(clusters))
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        batch_count = 0
        for position, cluster_index in enumerate(order, start=1):
            x_np, y_np, bad_np, oversize_bad_np = clusters[int(cluster_index)]
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            if bad_mass_weight > 0.0 and bad_np.any():
                bad = torch.as_tensor(bad_np, dtype=torch.bool, device=device)
                normalized = prediction / prediction.sum().clamp_min(1e-6)
                loss = loss + bad_mass_weight * normalized[bad].sum().square()
            if oversize_bad_mass_weight > 0.0 and oversize_bad_np.any():
                oversize_bad = torch.as_tensor(
                    oversize_bad_np, dtype=torch.bool, device=device
                )
                normalized = prediction / prediction.sum().clamp_min(1e-6)
                loss = (
                    loss
                    + oversize_bad_mass_weight
                    * normalized[oversize_bad].sum().square()
                )
            loss = loss / max(1, config.deep_sets_batch_clusters)
            loss.backward()
            epoch_losses.append(float(loss.detach().cpu()))
            batch_count += 1
            if (
                batch_count >= config.deep_sets_batch_clusters
                or position == len(order)
            ):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                batch_count = 0
        if epoch_losses:
            losses.append(float(np.mean(epoch_losses)))

    return {
        "model_type": "deep_sets",
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "feature_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "model_columns": model_columns,
        "config": asdict(config),
        "preprocess": preprocess,
        "input_dim": int(train_matrix.shape[1]),
        "train_rows": int(train_mask.sum()),
        "train_clusters": int(len(clusters)),
        "target_mean": float(target.loc[train_mask].mean()),
        "train_losses": losses,
    }


def _predict_deep_sets(features: pd.DataFrame, *, payload: dict[str, Any]) -> np.ndarray:
    config = CandidateSetWeightConfig(**payload.get("config", {}))
    numeric_columns = list(payload["feature_columns"])
    categorical_columns = list(payload.get("categorical_columns", ()))
    model_columns = list(payload.get("model_columns", numeric_columns))
    matrix = _deep_sets_matrix(
        features[model_columns],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        preprocess=payload["preprocess"],
    )
    device = _resolve_torch_device(config.deep_sets_device)
    model = _DeepSetsWeightRegressor(
        input_dim=int(payload["input_dim"]),
        hidden_dim=config.deep_sets_hidden_dim,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    output = np.zeros(len(features), dtype=float)
    cluster_ids = features["ranker_cluster_id"].astype(str).to_numpy()
    with torch.no_grad():
        for _, indices in _cluster_index_groups(cluster_ids):
            x = torch.as_tensor(matrix[indices], dtype=torch.float32, device=device)
            output[indices] = model(x).detach().cpu().numpy()
    return output


def _fit_deep_sets_preprocess(
    features: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    numeric = _unique_numeric_frame(features, numeric_columns)
    means = numeric.mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stds = numeric.std().replace([np.inf, -np.inf], np.nan).fillna(1.0)
    stds = stds.mask(stds <= 1e-8, 1.0)
    categories = {
        column: sorted(str(value) for value in features[column].fillna("").unique())
        for column in categorical_columns
    }
    return {
        "numeric_mean": means.to_dict(),
        "numeric_std": stds.to_dict(),
        "categories": categories,
    }


def _deep_sets_matrix(
    features: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    preprocess: dict[str, Any],
) -> np.ndarray:
    numeric = _unique_numeric_frame(features, numeric_columns)
    means = pd.Series(preprocess["numeric_mean"])
    stds = pd.Series(preprocess["numeric_std"])
    numeric = numeric.fillna(means)
    numeric = ((numeric - means) / stds).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    parts = [numeric.to_numpy(dtype=np.float32, copy=False)]
    for column in categorical_columns:
        values = features[column].fillna("").astype(str)
        categories = list(preprocess["categories"].get(column, ()))
        category_index = {category: index for index, category in enumerate(categories)}
        one_hot = np.zeros((len(features), len(categories)), dtype=np.float32)
        for row_index, value in enumerate(values):
            index = category_index.get(value)
            if index is not None:
                one_hot[row_index, index] = 1.0
        if one_hot.shape[1] > 0:
            parts.append(one_hot)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _unique_numeric_frame(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for column in columns:
        values = features.loc[:, column]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        data[column] = pd.to_numeric(values, errors="coerce")
    return pd.DataFrame(data, index=features.index)


def _deep_sets_clusters(
    matrix: np.ndarray,
    target: np.ndarray,
    true_iou: np.ndarray,
    oversize_area: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    bad_mass_margin: float,
    oversize_bad_mass_margin: float,
    oversize_area_log_ratio_threshold: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    clusters = []
    for _, indices in _cluster_index_groups(cluster_ids):
        y = target[indices].astype(np.float32, copy=False)
        iou = true_iou[indices].astype(np.float32, copy=False)
        area = oversize_area[indices].astype(np.float32, copy=False)
        valid = np.isfinite(y) & np.isfinite(iou)
        if not valid.any():
            continue
        valid_iou = iou[valid]
        valid_area = area[valid]
        oracle = np.nanmax(valid_iou)
        bad = valid_iou <= (oracle - float(bad_mass_margin))
        oversize_bad = (
            (valid_iou <= (oracle - float(oversize_bad_mass_margin)))
            & np.isfinite(valid_area)
            & (valid_area >= float(oversize_area_log_ratio_threshold))
        )
        clusters.append(
            (
                matrix[indices][valid],
                y[valid],
                bad.astype(bool),
                oversize_bad.astype(bool),
            )
        )
    return clusters


def _deep_sets_oversize_area_values(
    features: pd.DataFrame,
    *,
    config: CandidateSetWeightConfig,
) -> np.ndarray:
    column = config.deep_sets_oversize_area_column
    if column not in features.columns:
        return np.full(len(features), np.nan, dtype=float)
    return pd.to_numeric(features[column], errors="coerce").astype(float).to_numpy()


def _cluster_index_groups(cluster_ids: np.ndarray) -> list[tuple[str, np.ndarray]]:
    groups: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        groups.setdefault(str(cluster_id), []).append(index)
    return [
        (cluster_id, np.asarray(indices, dtype=int))
        for cluster_id, indices in groups.items()
    ]


def _resolve_torch_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_payload(model_payload: dict[str, Any] | Path) -> dict[str, Any]:
    if isinstance(model_payload, (str, Path)):
        return joblib.load(model_payload)
    return model_payload


def _output_path_for(path: Path, output_root: Path, relative_root: Path | None) -> Path:
    if relative_root is None:
        return output_root / path.name
    return output_root / path.relative_to(relative_root)
