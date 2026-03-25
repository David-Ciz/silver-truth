from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
import math

import mlflow
import pandas as pd
from mlflow.entities import Experiment
from mlflow.tracking import MlflowClient

# Anchor to the project root (two levels up from this file: src/silver_truth → project root)
# so the path resolves correctly regardless of the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MLFLOW_TRACKING_URI = str(_PROJECT_ROOT / "data" / "mlflow" / "mlruns")
DEFAULT_MLFLOW_ARTIFACT_ROOT = str(_PROJECT_ROOT / "data" / "mlflow" / "mlartifacts")

ABLATION_EXPERIMENT_ENV = "SILVER_TRUTH_MLFLOW_EXPERIMENT_NAME"
ABLATION_PARENT_RUN_ENV = "SILVER_TRUTH_MLFLOW_PARENT_RUN_ID"
ABLATION_TRACKING_URI_ENV = "SILVER_TRUTH_MLFLOW_TRACKING_URI"
MLFLOW_ARTIFACT_ROOT_ENV = "SILVER_TRUTH_MLFLOW_ARTIFACT_ROOT"
ABLATION_ROOT_RUN_ENV = "SILVER_TRUTH_ABLATION_ROOT_RUN_ID"
ABLATION_RUN_KEY_ENV = "SILVER_TRUTH_ABLATION_STATE_RUN_ID"
ABLATION_CONFIG_ENV = "SILVER_TRUTH_ABLATION_CONFIG_PATH"

MLFLOW_PARENT_RUN_TAG = "mlflow.parentRunId"


def _run_command(command: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        return output if output else None
    except Exception:
        return None


def normalize_mlflow_tracking_uri(uri: Optional[str]) -> str:
    resolved = uri or DEFAULT_MLFLOW_TRACKING_URI
    if resolved.startswith("file:") or "://" in resolved:
        return resolved
    return Path(resolved).expanduser().resolve().as_uri()


def resolve_mlflow_tracking_uri(uri: Optional[str] = None) -> str:
    return normalize_mlflow_tracking_uri(os.getenv(ABLATION_TRACKING_URI_ENV) or uri)


def uses_database_backend(tracking_uri: Optional[str]) -> bool:
    if not tracking_uri:
        return False
    normalized = tracking_uri.lower()
    return normalized.startswith(
        (
            "sqlite:",
            "postgresql:",
            "postgresql+",
            "mysql:",
            "mysql+",
            "mssql:",
            "mssql+",
        )
    )


def normalize_mlflow_artifact_root(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if path.startswith("file:") or "://" in path:
        return path
    return Path(path).expanduser().resolve().as_uri()


def resolve_mlflow_artifact_root(path: Optional[str] = None) -> Optional[str]:
    configured = os.getenv(MLFLOW_ARTIFACT_ROOT_ENV) or path
    if configured:
        return normalize_mlflow_artifact_root(configured)
    if uses_database_backend(resolve_mlflow_tracking_uri()):
        return normalize_mlflow_artifact_root(DEFAULT_MLFLOW_ARTIFACT_ROOT)
    return None


def ensure_mlflow_experiment(
    experiment_name: str,
    *,
    tracking_uri: Optional[str] = None,
    artifact_root: Optional[str] = None,
) -> Experiment:
    resolved_tracking_uri = resolve_mlflow_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment

    create_kwargs: dict[str, Any] = {}
    resolved_artifact_root = resolve_mlflow_artifact_root(artifact_root)
    if resolved_artifact_root is not None:
        create_kwargs["artifact_location"] = resolved_artifact_root

    experiment_id = client.create_experiment(experiment_name, **create_kwargs)
    created = client.get_experiment(experiment_id)
    if created is None:
        raise RuntimeError(f"Failed to create MLflow experiment: {experiment_name}")
    return created


def resolve_mlflow_experiment_name(name: Optional[str] = None) -> Optional[str]:
    return os.getenv(ABLATION_EXPERIMENT_ENV) or name


def get_inherited_parent_run_id() -> Optional[str]:
    return os.getenv(ABLATION_PARENT_RUN_ENV)


def get_ablation_context_tags(
    extra_tags: Optional[Mapping[str, object]] = None,
) -> dict[str, str]:
    tags: dict[str, str] = {}
    root_run_id = os.getenv(ABLATION_ROOT_RUN_ENV)
    state_run_id = os.getenv(ABLATION_RUN_KEY_ENV)
    config_path = os.getenv(ABLATION_CONFIG_ENV)
    if root_run_id:
        tags["ablation_root_run_id"] = root_run_id
    if state_run_id:
        tags["ablation_state_run_id"] = state_run_id
    if config_path:
        tags["ablation_config_path"] = config_path
    if extra_tags:
        for key, value in extra_tags.items():
            if value is None:
                continue
            tags[str(key)] = str(value)
    return tags


def start_managed_mlflow_run(
    *,
    run_id: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    nested: bool = False,
    parent_run_id: Optional[str] = None,
    tags: Optional[Mapping[str, object]] = None,
    description: Optional[str] = None,
    log_system_metrics: Optional[bool] = None,
):
    tracking_uri = resolve_mlflow_tracking_uri(mlflow_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = resolve_mlflow_experiment_name(mlflow_experiment)
    experiment_id: Optional[str] = None
    if experiment_name:
        experiment_id = ensure_mlflow_experiment(
            experiment_name,
            tracking_uri=tracking_uri,
        ).experiment_id

    effective_parent_run_id = parent_run_id
    if (
        effective_parent_run_id is None
        and run_id is None
        and mlflow.active_run() is None
    ):
        effective_parent_run_id = get_inherited_parent_run_id()

    merged_tags = get_ablation_context_tags(tags)
    if effective_parent_run_id and MLFLOW_PARENT_RUN_TAG not in merged_tags:
        merged_tags[MLFLOW_PARENT_RUN_TAG] = effective_parent_run_id

    kwargs: dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name,
        "nested": nested,
        "parent_run_id": effective_parent_run_id,
        "description": description,
        "log_system_metrics": log_system_metrics,
    }
    if experiment_id is not None:
        kwargs["experiment_id"] = experiment_id
    if merged_tags:
        kwargs["tags"] = merged_tags
    return mlflow.start_run(**kwargs)


def get_dvc_commit(repo_root: Optional[Path] = None) -> str:
    """
    Best-effort DVC state identifier.

    Priority:
    1) DVC/Git commit env vars.
    2) Current git commit hash (+dirty when DVC-tracked config changed).
    3) "unknown".
    """
    for env_key in ("DVC_COMMIT", "DVC_REV", "GIT_COMMIT", "CI_COMMIT_SHA"):
        value = os.getenv(env_key)
        if value:
            return value

    commit = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if not commit:
        return "unknown"

    dirty = _run_command(
        ["git", "status", "--porcelain", "--", "dvc.yaml", "dvc.lock", "params.yaml"],
        cwd=repo_root,
    )
    if dirty:
        return f"{commit}+dirty"
    return commit


def infer_dataset_name_from_text(values: Iterable[object]) -> str:
    pattern = re.compile(r"(BF-C2DL-HSC|BF-C2DL-MuSC|DIC-C2DH-HeLa)")
    for value in values:
        if value is None:
            continue
        text = str(value)
        match = pattern.search(text)
        if match:
            return match.group(1)
    return "unknown"


def infer_split_label(values: Iterable[object]) -> str:
    cleaned = sorted(
        {str(value) for value in values if value is not None and str(value)}
    )
    if not cleaned:
        return "unknown"
    if len(cleaned) == 1:
        return cleaned[0]
    return "multi:" + ",".join(cleaned)


def infer_split_from_dataframe(df: pd.DataFrame) -> str:
    if "split" not in df.columns:
        return "unknown"
    return infer_split_label(df["split"].dropna().tolist())


def set_common_mlflow_tags(
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    repo_root: Optional[Path] = None,
    extra_tags: Optional[dict[str, object]] = None,
) -> None:
    mlflow.set_tag("dvc_commit", get_dvc_commit(repo_root))
    if dataset:
        mlflow.set_tag("dataset", dataset)
    if split:
        mlflow.set_tag("split", split)
    if extra_tags:
        for key, value in extra_tags.items():
            if value is None:
                continue
            mlflow.set_tag(key, str(value))


def set_ablation_tags(
    dataset: str,
    fold: str,
    phase: str,
    step: str,
    variant: str,
    ablation_mode: Optional[str] = None,
    qa_threshold: Optional[float] = None,
    repo_root: Optional[Path] = None,
) -> None:
    """
    Set standardised MLflow tags for ablation experiment runs.

    Naming convention for the MLflow experiment (set *before* starting the run):
        ``{phase}-{dataset}-{variant}-{fold}``
        e.g. ``phaseC-BF-C2DL-HSC-baseline-fold1``

    Every run tagged this way is filterable by any axis in the MLflow Compare Runs UI.

    Parameters
    ----------
    dataset:       dataset name, e.g. ``BF-C2DL-HSC``
    fold:          fold identifier, e.g. ``fold-1``
    phase:         pipeline phase, e.g. ``phaseA``, ``phaseB``, ``phaseC``
    step:          step name within the phase, e.g. ``competitor_baseline``, ``qa_train``
    variant:       experiment variant name, e.g. ``baseline``, ``resnet18_qa``
    ablation_mode: ablation mode for Phase C steps, e.g. ``full_pipeline``, ``qa_only``
    qa_threshold:  QA filter threshold if applicable
    repo_root:     project root for git/dvc commit resolution
    """
    mlflow.set_tag("dataset", dataset)
    mlflow.set_tag("fold", fold)
    mlflow.set_tag("phase", phase)
    mlflow.set_tag("step", step)
    mlflow.set_tag("variant", variant)
    mlflow.set_tag("dvc_commit", get_dvc_commit(repo_root))
    if ablation_mode is not None:
        mlflow.set_tag("ablation_mode", ablation_mode)
    if qa_threshold is not None:
        mlflow.set_tag("qa_threshold", str(qa_threshold))


def canonical_split_name(split: str) -> str:
    split_lower = str(split).strip().lower()
    if split_lower in {"val", "valid"}:
        return "validation"
    return split_lower


def metric_split_alias(split: str) -> str:
    canonical = canonical_split_name(split)
    if canonical == "validation":
        return "val"
    return canonical


def _is_finite_number(value: object) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _to_finite_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def set_evaluation_tags(
    pipeline_family: str,
    evaluation_level: str,
    setup_name: Optional[str] = None,
    variant: Optional[str] = None,
    fold: Optional[str] = None,
    qa_mode: Optional[str] = None,
    qa_threshold: Optional[float] = None,
    source_experiment: Optional[str] = None,
    source_run_id: Optional[str] = None,
    extra_tags: Optional[dict[str, object]] = None,
) -> None:
    mlflow.set_tag("pipeline_family", pipeline_family)
    mlflow.set_tag("evaluation_level", evaluation_level)
    mlflow.set_tag("metric_schema_version", "paper_eval_v1")
    mlflow.set_tag("primary_metrics", "test_iou,test_f1")
    if setup_name:
        mlflow.set_tag("setup_name", setup_name)
    if variant:
        mlflow.set_tag("variant", variant)
    if fold:
        mlflow.set_tag("fold", fold)
    if qa_mode:
        mlflow.set_tag("qa_mode", qa_mode)
    if qa_threshold is not None:
        mlflow.set_tag("qa_threshold", str(qa_threshold))
    if source_experiment:
        mlflow.set_tag("source_experiment", source_experiment)
    if source_run_id:
        mlflow.set_tag("source_run_id", source_run_id)
    if extra_tags:
        for key, value in extra_tags.items():
            if value is None:
                continue
            mlflow.set_tag(key, str(value))


def log_standardized_split_metrics(
    split_metrics: dict[str, object],
    count_metric_name: str = "eval_count",
) -> dict[str, float]:
    """
    Log canonical split metrics and return the emitted metric dict.

    Expected input keys follow the project's existing convention, e.g.:
    ``test_mean_jaccard``, ``validation_mean_f1``, ``overall_count``.
    """
    emitted: dict[str, float] = {}
    for split in ("train", "validation", "test", "overall"):
        split_alias = metric_split_alias(split)
        jaccard_key = f"{split}_mean_jaccard"
        f1_key = f"{split}_mean_f1"
        count_key = f"{split}_count"

        jaccard_value = _to_finite_float(split_metrics.get(jaccard_key))
        if jaccard_value is not None:
            emitted[f"{split_alias}_iou"] = jaccard_value
        f1_value = _to_finite_float(split_metrics.get(f1_key))
        if f1_value is not None:
            emitted[f"{split_alias}_f1"] = f1_value
        test_count_value = _to_finite_float(split_metrics.get(count_key))
        if split == "test" and test_count_value is not None:
            emitted[count_metric_name] = test_count_value

    if emitted:
        mlflow.log_metrics(emitted)
    return emitted


def log_standardized_single_split_metrics(
    split: str,
    iou: Optional[float],
    f1: Optional[float],
    count: Optional[int] = None,
    count_metric_name: str = "eval_count",
) -> dict[str, float]:
    """Log canonical metrics for a single evaluated split, e.g. a final test run."""
    split_alias = metric_split_alias(split)
    emitted: dict[str, float] = {}
    iou_value = _to_finite_float(iou)
    if iou_value is not None:
        emitted[f"{split_alias}_iou"] = iou_value
    f1_value = _to_finite_float(f1)
    if f1_value is not None:
        emitted[f"{split_alias}_f1"] = f1_value
    count_value = _to_finite_float(count)
    if count_value is not None:
        emitted[count_metric_name] = count_value
    if emitted:
        mlflow.log_metrics(emitted)
    return emitted
