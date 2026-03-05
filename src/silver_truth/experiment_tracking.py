from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import mlflow
import pandas as pd

DEFAULT_MLFLOW_TRACKING_URI = "data/mlflow/mlruns"


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
