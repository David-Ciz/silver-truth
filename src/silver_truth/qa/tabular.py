from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from silver_truth.experiment_tracking import (
    DEFAULT_MLFLOW_TRACKING_URI,
    MLFLOW_PARENT_RUN_TAG,
    ensure_mlflow_experiment,
    get_ablation_context_tags,
    get_inherited_parent_run_id,
    resolve_mlflow_experiment_name,
    resolve_mlflow_tracking_uri,
    start_managed_mlflow_run,
)
from silver_truth.metrics.qa_model_evaluation import (
    calculate_regression_metrics,
    calculate_tolerance_accuracy,
)
from silver_truth.qa.cnn import (
    JaccardDataset,
    get_split_indices,
    save_results_to_excel,
    set_seed,
)


def _build_feature_matrix(
    dataset: JaccardDataset,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    targets = []
    cell_ids = []
    for idx in indices:
        image_path = dataset.image_paths[idx]
        feature_vector = dataset._get_cached_metadata(idx, image_path)
        features.append(np.asarray(feature_vector, dtype=np.float32))
        targets.append(float(dataset.targets[idx]))
        cell_ids.append(dataset.cell_ids[idx])

    if not features:
        feature_dim = len(dataset.metadata_features)
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    return (
        np.stack(features, axis=0).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        cell_ids,
    )


def _build_regressor(model_type: str, seed: int):
    normalized = str(model_type).strip().lower()
    if normalized == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            min_samples_leaf=8,
            l2_regularization=0.01,
            random_state=seed,
        )
    if normalized == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(
        f"Unsupported tabular QA model_type: {model_type}. "
        "Supported: hist_gradient_boosting, random_forest."
    )


def _predict_clipped(model, features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.empty((0,), dtype=np.float32)
    predictions = np.asarray(model.predict(features), dtype=np.float32)
    return np.clip(predictions, 0.0, 1.0)


def save_model(model, path: Path, metadata: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, path)
    print(f"Model saved to {path}")


def load_model(path: Path):
    payload = joblib.load(path)
    return payload["model"], payload.get("metadata", {})


def _results_dataframe(
    *,
    cell_ids: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "Jaccard index": targets,
            "Predicted Jaccard index": predictions,
        }
    )


def _evaluate_split(
    *,
    model,
    dataset: JaccardDataset,
    indices: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    features, targets, cell_ids = _build_feature_matrix(dataset, indices)
    predictions = _predict_clipped(model, features)
    results_df = _results_dataframe(
        cell_ids=cell_ids,
        targets=targets,
        predictions=predictions,
    )
    metrics = {}
    if len(targets) > 0:
        metrics = {**calculate_tolerance_accuracy(targets, predictions)}
        if len(targets) >= 2:
            metrics.update(calculate_regression_metrics(targets, predictions))
        else:
            residuals = predictions - targets
            metrics.update(
                {
                    "mae": float(np.mean(np.abs(residuals))),
                    "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
                    "mse": float(np.mean(np.square(residuals))),
                    "mean_residual": float(np.mean(residuals)),
                    "std_residual": float(np.std(residuals)),
                    "n_samples": float(len(targets)),
                }
            )
    return results_df, metrics


def train(
    parquet_file,
    data_root=None,
    target_column=None,
    metadata_features: Optional[Sequence[str] | str] = "default",
    output_model="qa_tabular.joblib",
    output_excel="results_tabular.xlsx",
    model_type="hist_gradient_boosting",
    seed=42,
    mlflow_tracking_uri=DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_experiment="qa-tabular",
    mlflow_run_name=None,
):
    set_seed(seed)
    print(f"Random seed: {seed}")

    output_model_path = Path(output_model).expanduser()
    output_excel_path = Path(output_excel).expanduser()
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_tracking_uri = resolve_mlflow_tracking_uri(mlflow_tracking_uri)
    resolved_experiment_name = (
        resolve_mlflow_experiment_name(mlflow_experiment) or mlflow_experiment
    )
    inherited_parent_run_id = get_inherited_parent_run_id()
    logger_tags = get_ablation_context_tags()
    run_id: Optional[str] = None

    if inherited_parent_run_id or logger_tags:
        mlflow.set_tracking_uri(resolved_tracking_uri)
        experiment = ensure_mlflow_experiment(
            resolved_experiment_name,
            tracking_uri=resolved_tracking_uri,
        )
        create_tags = dict(logger_tags)
        if inherited_parent_run_id:
            create_tags[MLFLOW_PARENT_RUN_TAG] = inherited_parent_run_id
        run = MlflowClient(tracking_uri=resolved_tracking_uri).create_run(
            experiment_id=experiment.experiment_id,
            run_name=mlflow_run_name,
            tags=create_tags or None,
        )
        run_id = run.info.run_id

    dataset = JaccardDataset(
        parquet_file,
        data_root=data_root,
        transform=None,
        augment=False,
        target_column=target_column,
        metadata_features=metadata_features,
        preload_images=True,
    )
    if not dataset.metadata_features:
        raise ValueError(
            "Tabular QA requires metadata_features. "
            "Use --metadata-features default or a non-empty feature list."
        )

    print(f"Using target column: {dataset.target_column}")
    print("Using metadata features: " + ",".join(dataset.metadata_features))

    train_indices, val_indices, test_indices = get_split_indices(dataset)
    print(
        f"Dataset splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    train_features, train_targets, _ = _build_feature_matrix(dataset, train_indices)
    model = _build_regressor(model_type, seed)
    model.fit(train_features, train_targets)

    train_results, train_metrics = _evaluate_split(
        model=model,
        dataset=dataset,
        indices=train_indices,
    )
    val_results, val_metrics = _evaluate_split(
        model=model,
        dataset=dataset,
        indices=val_indices,
    )
    test_results, test_metrics = _evaluate_split(
        model=model,
        dataset=dataset,
        indices=test_indices,
    )

    save_results_to_excel(
        train_results,
        val_results,
        test_results,
        str(output_excel_path),
    )

    metadata = {
        "backend": "tabular",
        "parquet_file": str(parquet_file),
        "model_type": model_type,
        "seed": seed,
        "target_column": str(dataset.target_column),
        "metadata_features": ",".join(dataset.metadata_features),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
    }
    save_model(model, output_model_path, metadata)

    split_metric_payload = {}
    for split_name, metrics in (
        ("train", train_metrics),
        ("validation", val_metrics),
        ("test", test_metrics),
    ):
        for key, value in metrics.items():
            split_metric_payload[f"{split_name}_{key}"] = value

    mlflow.set_tracking_uri(resolved_tracking_uri)
    with start_managed_mlflow_run(
        run_id=run_id,
        mlflow_tracking_uri=resolved_tracking_uri,
        mlflow_experiment=resolved_experiment_name,
        run_name=mlflow_run_name,
    ):
        mlflow.log_params(
            {
                "qa_backend": "tabular",
                "model_type": model_type,
                "seed": seed,
                "parquet_file": str(parquet_file),
                "target_column": str(dataset.target_column),
                "metadata_features": ",".join(dataset.metadata_features),
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
            }
        )
        if split_metric_payload:
            mlflow.log_metrics(split_metric_payload)
        mlflow.log_artifact(str(output_model_path))
        mlflow.log_artifact(str(output_excel_path))


def evaluate(
    parquet_file,
    data_root=None,
    target_column=None,
    metadata_features: Optional[Sequence[str] | str] = None,
    model_path=None,
    output_excel="results_tabular.xlsx",
):
    if model_path is None:
        raise ValueError("model_path is required for evaluation.")

    model, metadata = load_model(Path(model_path))
    print(f"Model metadata: {metadata}")

    effective_target_column = (
        target_column if target_column is not None else metadata.get("target_column")
    )
    effective_metadata_features = (
        metadata_features
        if metadata_features is not None
        else metadata.get("metadata_features")
    )

    dataset = JaccardDataset(
        parquet_file,
        data_root=data_root,
        transform=None,
        augment=False,
        target_column=effective_target_column,
        metadata_features=effective_metadata_features,
        preload_images=True,
    )
    print(f"Using target column: {dataset.target_column}")
    print("Using metadata features: " + ",".join(dataset.metadata_features))

    train_indices, val_indices, test_indices = get_split_indices(dataset)
    train_results, _ = _evaluate_split(model=model, dataset=dataset, indices=train_indices)
    val_results, _ = _evaluate_split(model=model, dataset=dataset, indices=val_indices)
    test_results, _ = _evaluate_split(model=model, dataset=dataset, indices=test_indices)
    save_results_to_excel(train_results, val_results, test_results, output_excel)
