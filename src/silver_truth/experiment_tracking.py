from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import mlflow
import pandas as pd

# Anchor to the project root (two levels up from this file: src/silver_truth → project root)
# so the path resolves correctly regardless of the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MLFLOW_TRACKING_URI = str(_PROJECT_ROOT / "data" / "mlflow" / "mlruns")


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
