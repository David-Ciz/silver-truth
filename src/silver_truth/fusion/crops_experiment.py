from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import tifffile

from silver_truth.experiment_tracking import (
    DEFAULT_MLFLOW_TRACKING_URI,
    infer_dataset_name_from_text,
    infer_split_from_dataframe,
    set_common_mlflow_tags,
)
from silver_truth.fusion.fusion import FusionModel, fuse_segmentations
from silver_truth.metrics.evaluation_logic import evaluate_by_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JAR_PATH = (
    PROJECT_ROOT
    / "src/silver_truth/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar"
)
MLFLOW_TRACKING_PATH = DEFAULT_MLFLOW_TRACKING_URI
DEFAULT_FUSION_THRESHOLD = 1.0
WEIGHT_COLUMN_CANDIDATES = [
    "fusion_weight",
    "competitor_weight",
    "weight",
]

FLAT_MODELS = [
    "THRESHOLD_FLAT",
    "MAJORITY_FLAT",
    "SIMPLE",
    "BIC_FLAT_VOTING",
]
USER_WEIGHTED_MODELS = [
    "THRESHOLD_USER",
    "BIC_WEIGHTED_VOTING",
]
JOB_WEIGHT_FORMAT_MODELS = [
    "THRESHOLD_USER",
    "BIC_FLAT_VOTING",
    "BIC_WEIGHTED_VOTING",
]
ALL_MODELS = FLAT_MODELS + USER_WEIGHTED_MODELS

REQUIRED_COLUMNS = [
    "campaign_number",
    "original_image_key",
    "label",
    "competitor",
    "stacked_path",
]

CellLabel = Union[int, str]
CellKey = Tuple[str, str, CellLabel]
CellGroups = Dict[CellKey, Dict[str, Path]]
CellSplits = Dict[CellKey, str]
CellPathLookup = Dict[CellKey, Optional[str]]


def select_models(
    models: Sequence[str],
    all_models: bool = False,
    flat_models_only: bool = False,
) -> List[str]:
    """Resolve model list from CLI flags."""
    if all_models:
        selected = ALL_MODELS
    elif flat_models_only:
        selected = FLAT_MODELS
    elif models:
        selected = [model.upper() for model in models]
    else:
        raise ValueError(
            "Specify --models, --all-models, or --flat-models-only to run fusion."
        )

    invalid = sorted(set(selected) - set(ALL_MODELS))
    if invalid:
        raise ValueError(f"Unsupported fusion model(s): {', '.join(invalid)}")

    # De-duplicate while preserving order
    deduped = list(dict.fromkeys(selected))
    return deduped


def _normalize_campaign(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    as_text = str(value).strip()
    if as_text.isdigit():
        return f"{int(as_text):02d}"
    return as_text


def _normalize_label(value: Any) -> CellLabel:
    if pd.isna(value):
        return "nan"
    try:
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
    except (TypeError, ValueError):
        pass
    return str(value)


def build_competitor_dir_names(competitors: Sequence[str]) -> Dict[str, str]:
    used: set[str] = set()
    mapping: Dict[str, str] = {}
    for competitor in competitors:
        base_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", competitor).strip("_")
        if not base_name:
            base_name = "competitor"

        candidate = base_name
        suffix = 2
        while candidate in used:
            candidate = f"{base_name}_{suffix}"
            suffix += 1

        used.add(candidate)
        mapping[competitor] = candidate

    return mapping


def _resolve_stacked_path(path_value: Any, qa_parquet: Path) -> Optional[Path]:
    if pd.isna(path_value):
        return None

    candidate = Path(str(path_value)).expanduser()
    if candidate.is_absolute():
        return candidate

    project_candidate = PROJECT_ROOT / candidate
    parquet_relative_candidate = qa_parquet.parent / candidate

    if project_candidate.exists():
        return project_candidate
    if parquet_relative_candidate.exists():
        return parquet_relative_candidate
    return project_candidate


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def build_cell_groups(
    df: pd.DataFrame, qa_parquet: Path
) -> Tuple[CellGroups, CellSplits, Dict[CellKey, Dict[str, Any]]]:
    """Group rows by logical cell key and collect stacked paths per competitor."""
    has_split = "split" in df.columns
    has_gt_image = "gt_image" in df.columns
    has_crop_bounds = all(
        column in df.columns
        for column in ["crop_y_start", "crop_y_end", "crop_x_start", "crop_x_end"]
    )

    columns = REQUIRED_COLUMNS.copy()
    if has_split:
        columns.append("split")
    if has_gt_image:
        columns.append("gt_image")
    if has_crop_bounds:
        columns.extend(["crop_y_start", "crop_y_end", "crop_x_start", "crop_x_end"])
    grouped_rows = df[columns]

    cell_groups: CellGroups = {}
    cell_splits: CellSplits = {}
    cell_metadata: Dict[CellKey, Dict[str, Any]] = {}
    unresolved_paths = 0

    for row in grouped_rows.itertuples(index=False, name="CellRow"):
        campaign_number = _normalize_campaign(row.campaign_number)
        original_image_key = str(row.original_image_key)
        label = _normalize_label(row.label)
        competitor = str(row.competitor)
        resolved_path = _resolve_stacked_path(row.stacked_path, qa_parquet)

        key: CellKey = (campaign_number, original_image_key, label)
        if key not in cell_groups:
            cell_groups[key] = {}

        metadata = cell_metadata.setdefault(key, {})
        if has_gt_image:
            gt_resolved_path = _resolve_stacked_path(
                getattr(row, "gt_image"), qa_parquet
            )
            if gt_resolved_path is not None:
                metadata.setdefault("gt_image_path", gt_resolved_path)
        if has_crop_bounds:
            y_start = getattr(row, "crop_y_start")
            y_end = getattr(row, "crop_y_end")
            x_start = getattr(row, "crop_x_start")
            x_end = getattr(row, "crop_x_end")
            if (
                pd.notna(y_start)
                and pd.notna(y_end)
                and pd.notna(x_start)
                and pd.notna(x_end)
            ):
                metadata.setdefault("crop_y_start", int(y_start))
                metadata.setdefault("crop_y_end", int(y_end))
                metadata.setdefault("crop_x_start", int(x_start))
                metadata.setdefault("crop_x_end", int(x_end))

        if resolved_path is None:
            unresolved_paths += 1
            continue

        cell_groups[key][competitor] = resolved_path

        if has_split:
            split_value = str(getattr(row, "split"))
            if split_value and split_value.lower() != "nan":
                cell_splits.setdefault(key, split_value)

    if unresolved_paths:
        logger.warning(
            "Skipped %d row(s) with missing stacked_path while building cell groups.",
            unresolved_paths,
        )

    # Drop keys that ended up with no competitor files
    empty_keys = [key for key, value in cell_groups.items() if not value]
    for key in empty_keys:
        cell_groups.pop(key, None)
        cell_splits.pop(key, None)
        cell_metadata.pop(key, None)

    return cell_groups, cell_splits, cell_metadata


def _binary_mask(layer: np.ndarray) -> np.ndarray:
    return (layer > 0).astype(np.uint8)


def _label_specific_mask(segmentation: np.ndarray, label: CellLabel) -> np.ndarray:
    numeric_label: Optional[int] = None
    if isinstance(label, int):
        numeric_label = label
    elif isinstance(label, str) and label.isdigit():
        numeric_label = int(label)

    if numeric_label is not None and numeric_label > 0:
        label_mask = (segmentation == numeric_label).astype(np.uint8)
        if np.any(label_mask):
            return label_mask
    return _binary_mask(segmentation)


def _crop_to_shape(source: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    cropped = np.zeros(target_shape, dtype=source.dtype)
    y_size = min(target_shape[0], source.shape[0])
    x_size = min(target_shape[1], source.shape[1])
    cropped[:y_size, :x_size] = source[:y_size, :x_size]
    return cropped


def _gt_mask_from_metadata(
    metadata: Dict[str, Any],
    label: CellLabel,
    reference_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    gt_image_path = metadata.get("gt_image_path")
    if gt_image_path is None:
        return None

    y_start = metadata.get("crop_y_start")
    y_end = metadata.get("crop_y_end")
    x_start = metadata.get("crop_x_start")
    x_end = metadata.get("crop_x_end")
    if None in (y_start, y_end, x_start, x_end):
        return None
    assert y_start is not None
    assert y_end is not None
    assert x_start is not None
    assert x_end is not None

    gt_path = Path(gt_image_path)
    if not gt_path.exists():
        return None

    try:
        gt_full = tifffile.imread(gt_path)
    except Exception as exc:  # pragma: no cover - depends on source files
        logger.warning("Failed to read GT image %s: %s", gt_path, exc)
        return None

    if gt_full.ndim > 2:
        gt_full = np.squeeze(gt_full)
    if gt_full.ndim != 2:
        return None

    ys, ye = int(y_start), int(y_end)
    xs, xe = int(x_start), int(x_end)
    gt_crop = gt_full[ys:ye]
    gt_crop = gt_crop[:, xs:xe]
    if gt_crop.size == 0:
        return None

    if gt_crop.shape != reference_shape:
        gt_crop = _crop_to_shape(gt_crop, reference_shape)

    return _label_specific_mask(gt_crop, label)


def prepare_fusion_job(
    job_dir: Path,
    cell_groups: CellGroups,
    competitors: Sequence[str],
    cell_metadata: Optional[Dict[CellKey, Dict[str, Any]]] = None,
    competitor_dir_names: Optional[Dict[str, str]] = None,
) -> Dict[int, CellKey]:
    """
    Build a fusion job directory in which each logical cell is represented as a
    synthetic timepoint.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = job_dir / "GT"
    tra_dir = job_dir / "TRA"
    gt_dir.mkdir(parents=True, exist_ok=True)
    tra_dir.mkdir(parents=True, exist_ok=True)

    for competitor in competitors:
        competitor_dir_name = (
            competitor_dir_names.get(competitor, competitor)
            if competitor_dir_names
            else competitor
        )
        (job_dir / competitor_dir_name).mkdir(parents=True, exist_ok=True)

    mapping: Dict[int, CellKey] = {}
    skipped_cells = 0
    gt_from_stack = 0
    gt_from_metadata = 0
    gt_empty = 0
    tra_from_stack = 0
    tra_from_gt = 0
    tra_from_union = 0

    for key in sorted(cell_groups.keys()):
        competitor_paths = cell_groups[key]
        loaded_stacks: Dict[str, np.ndarray] = {}

        for competitor in competitors:
            source_path = competitor_paths.get(competitor)
            if source_path is None or not source_path.exists():
                continue
            try:
                stack = tifffile.imread(source_path)
            except Exception as exc:  # pragma: no cover - depends on source files
                logger.warning("Failed to read %s: %s", source_path, exc)
                continue

            if stack.ndim != 3 or stack.shape[0] < 2:
                logger.warning(
                    "Unexpected stack shape %s for %s", stack.shape, source_path
                )
                continue

            loaded_stacks[competitor] = stack

        if not loaded_stacks:
            skipped_cells += 1
            continue

        idx = len(mapping)
        mapping[idx] = key
        file_name = f"mask{idx:04d}.tif"

        reference_stack = next(iter(loaded_stacks.values()))
        reference_shape = reference_stack[1].shape

        for competitor in competitors:
            stack_opt = loaded_stacks.get(competitor)
            if stack_opt is None or stack_opt[1].shape != reference_shape:
                seg_binary = np.zeros(reference_shape, dtype=np.uint8)
            else:
                seg_binary = _binary_mask(stack_opt[1])
            competitor_dir_name = (
                competitor_dir_names.get(competitor, competitor)
                if competitor_dir_names
                else competitor
            )
            tifffile.imwrite(job_dir / competitor_dir_name / file_name, seg_binary)

        metadata = cell_metadata.get(key, {}) if cell_metadata else {}

        gt_source = next(
            (stack for stack in loaded_stacks.values() if stack.shape[0] >= 3), None
        )
        if gt_source is None:
            gt_binary = _gt_mask_from_metadata(metadata, key[2], reference_shape)
            if gt_binary is None:
                gt_binary = np.zeros(reference_shape, dtype=np.uint8)
                gt_empty += 1
            else:
                gt_from_metadata += 1
        else:
            gt_binary = _binary_mask(gt_source[2])
            gt_from_stack += 1
        tifffile.imwrite(gt_dir / file_name, gt_binary)

        tra_source = next(
            (stack for stack in loaded_stacks.values() if stack.shape[0] >= 4), None
        )
        if tra_source is None:
            # The Java fusion tool expects a marker image. Use GT when available,
            # otherwise fallback to the union of competitor segmentations.
            if np.any(gt_binary):
                tra_binary = gt_binary.copy()
                tra_from_gt += 1
            else:
                seg_union = np.zeros(reference_shape, dtype=np.uint8)
                for stack in loaded_stacks.values():
                    if stack.shape[0] >= 2:
                        seg_union = np.maximum(seg_union, _binary_mask(stack[1]))
                tra_binary = seg_union
                tra_from_union += 1
        else:
            tra_binary = _binary_mask(tra_source[3])
            tra_from_stack += 1
        tifffile.imwrite(tra_dir / file_name, tra_binary)

    if skipped_cells:
        logger.warning(
            "Skipped %d cell(s) with no readable competitor stacks.", skipped_cells
        )
    logger.info(
        "GT sources: stack=%d metadata=%d empty=%d | TRA sources: stack=%d gt_fallback=%d union_fallback=%d",
        gt_from_stack,
        gt_from_metadata,
        gt_empty,
        tra_from_stack,
        tra_from_gt,
        tra_from_union,
    )
    logger.info("Prepared fusion job with %d cells.", len(mapping))
    return mapping


def write_job_file(
    job_dir: Path,
    competitors: Sequence[str],
    weighted: bool,
    competitor_dir_names: Optional[Dict[str, str]] = None,
    competitor_weights: Optional[Dict[str, float]] = None,
) -> Path:
    file_name = "job_file_with_weights.txt" if weighted else "job_file.txt"
    job_file_path = job_dir / file_name

    with open(job_file_path, "w", encoding="utf-8") as file:
        for competitor in competitors:
            competitor_dir_name = (
                competitor_dir_names.get(competitor, competitor)
                if competitor_dir_names
                else competitor
            )
            competitor_dir = (job_dir / competitor_dir_name).resolve()
            suffix = ""
            if weighted:
                if competitor_weights is None or competitor not in competitor_weights:
                    raise ValueError(
                        f"Missing weight for competitor '{competitor}' while writing weighted job file."
                    )
                suffix = f" {float(competitor_weights[competitor]):.6g}"
            file.write(f"{competitor_dir}/maskTTTT.tif{suffix}\n")
        tra_dir = (job_dir / "TRA").resolve()
        file.write(f"{tra_dir}/maskTTTT.tif\n")

    return job_file_path


def chunk_indices(indices: Sequence[int], chunk_size: int) -> List[List[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [
        list(indices[i : i + chunk_size]) for i in range(0, len(indices), chunk_size)
    ]


def _first_non_null_value(values: pd.Series) -> Any:
    for value in values:
        if pd.notna(value):
            return value
    return np.nan


def _build_key_path_lookup(
    mapping: Dict[int, CellKey],
    base_dir: Path,
    file_prefix: str,
) -> CellPathLookup:
    lookup: CellPathLookup = {}
    for idx, key in mapping.items():
        path = base_dir / f"{file_prefix}{idx:04d}.tif"
        lookup[key] = str(path) if path.exists() else None
    return lookup


def _enrich_qa_with_paths(
    qa_df: pd.DataFrame,
    fused_lookup: CellPathLookup,
    gt_lookup: CellPathLookup,
    fused_column: str,
) -> pd.DataFrame:
    enriched = qa_df.copy()
    cell_keys = list(
        zip(
            enriched["campaign_number"].map(_normalize_campaign),
            enriched["original_image_key"].astype(str),
            enriched["label"].map(_normalize_label),
        )
    )
    enriched[fused_column] = [fused_lookup.get(key) for key in cell_keys]
    enriched["crop_gt_path"] = [gt_lookup.get(key) for key in cell_keys]
    return enriched


def _build_cell_level_eval_df(
    qa_df_with_paths: pd.DataFrame,
    fused_column: str,
) -> pd.DataFrame:
    required = {
        "campaign_number",
        "original_image_key",
        "label",
        "crop_gt_path",
        fused_column,
    }
    missing = required - set(qa_df_with_paths.columns)
    if missing:
        raise ValueError(
            "Missing required column(s) for evaluation dataframe: "
            + ", ".join(sorted(missing))
        )

    working = qa_df_with_paths.copy()
    working["_campaign_norm"] = working["campaign_number"].map(_normalize_campaign)
    working["_original_image_key_norm"] = working["original_image_key"].astype(str)
    working["_label_norm"] = working["label"].map(_normalize_label)

    group_columns = ["_campaign_norm", "_original_image_key_norm", "_label_norm"]
    aggregate: Dict[str, Any] = {
        "campaign_number": _first_non_null_value,
        "original_image_key": _first_non_null_value,
        "label": _first_non_null_value,
        "crop_gt_path": _first_non_null_value,
        fused_column: _first_non_null_value,
    }
    if "split" in working.columns:
        aggregate["split"] = _first_non_null_value

    cell_df = (
        working.groupby(group_columns, as_index=False, dropna=False)
        .agg(aggregate)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return cell_df


def _persist_gt_masks(
    source_gt_dir: Path, destination_dir: Path, mapping: Dict[int, CellKey]
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for idx in mapping:
        source_path = source_gt_dir / f"mask{idx:04d}.tif"
        destination_path = destination_dir / f"mask{idx:04d}.tif"
        if not source_path.exists():
            logger.warning(
                "Missing staged GT file for index %04d: %s", idx, source_path
            )
            continue
        shutil.copy2(source_path, destination_path)


def _compute_metrics(fused_path: Path, gt_path: Path) -> Tuple[float, float]:
    fused = tifffile.imread(fused_path)
    gt = tifffile.imread(gt_path)

    fused_bin = fused > 0
    gt_bin = gt > 0

    tp = int(np.logical_and(fused_bin, gt_bin).sum())
    fp = int(np.logical_and(fused_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~fused_bin, gt_bin).sum())

    jaccard_denominator = tp + fp + fn
    if jaccard_denominator == 0:
        jaccard = 1.0
    else:
        jaccard = tp / jaccard_denominator

    f1_denominator = (2 * tp) + fp + fn
    if f1_denominator == 0:
        f1 = 1.0
    else:
        f1 = (2 * tp) / f1_denominator

    return float(jaccard), float(f1)


def inspect_fused_outputs(
    fusion_out_dir: Path,
    mapping: Dict[int, CellKey],
) -> Dict[str, float]:
    present_outputs = 0
    missing_outputs = 0
    nonempty_outputs = 0
    unreadable_outputs = 0

    for idx in mapping:
        fused_path = fusion_out_dir / f"fused_{idx:04d}.tif"
        if not fused_path.exists():
            missing_outputs += 1
            continue

        present_outputs += 1
        try:
            fused = tifffile.imread(fused_path)
            if np.any(fused > 0):
                nonempty_outputs += 1
        except Exception as exc:  # pragma: no cover - file quality dependent
            unreadable_outputs += 1
            logger.warning("Failed to read fused output %s: %s", fused_path, exc)

    sample_index = 6 if 6 in mapping else (min(mapping) if mapping else -1)
    sample_exists = 0.0
    sample_nonempty = 0.0
    if sample_index >= 0:
        sample_path = fusion_out_dir / f"fused_{sample_index:04d}.tif"
        if sample_path.exists():
            sample_exists = 1.0
            try:
                sample_nonempty = (
                    1.0 if np.any(tifffile.imread(sample_path) > 0) else 0.0
                )
            except Exception as exc:  # pragma: no cover - file quality dependent
                logger.warning(
                    "Failed to read sample fused output %s: %s", sample_path, exc
                )

    return {
        "expected_outputs": float(len(mapping)),
        "present_outputs": float(present_outputs),
        "missing_fused_outputs": float(missing_outputs),
        "nonempty_fused_outputs": float(nonempty_outputs),
        "unreadable_fused_outputs": float(unreadable_outputs),
        "sample_0006_exists": sample_exists,
        "sample_0006_nonempty": sample_nonempty,
    }


def evaluate_model_results(
    fusion_out_dir: Path,
    gt_dir: Path,
    mapping: Dict[int, CellKey],
    cell_splits: CellSplits,
    model: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, key in mapping.items():
        campaign, original_image_key, label = key
        fused_path = fusion_out_dir / f"fused_{idx:04d}.tif"
        gt_path = gt_dir / f"mask{idx:04d}.tif"

        fused_exists = fused_path.exists()
        gt_exists = gt_path.exists()

        fused_nonempty = np.nan
        if fused_exists:
            try:
                fused_nonempty = float(np.any(tifffile.imread(fused_path) > 0))
            except Exception as exc:  # pragma: no cover - file quality dependent
                logger.warning("Failed to inspect %s: %s", fused_path, exc)

        jaccard = np.nan
        f1 = np.nan
        if fused_exists and gt_exists:
            try:
                jaccard, f1 = _compute_metrics(fused_path, gt_path)
            except Exception as exc:  # pragma: no cover - file quality dependent
                logger.warning("Failed to evaluate %s: %s", fused_path, exc)

        rows.append(
            {
                "model": model,
                "campaign": campaign,
                "original_image_key": original_image_key,
                "label": label,
                "split": cell_splits.get(key),
                "fused_path": str(fused_path),
                "gt_path": str(gt_path),
                "fused_exists": fused_exists,
                "gt_exists": gt_exists,
                "fused_nonempty": fused_nonempty,
                "jaccard": jaccard,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


def summarize_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    valid = results_df.dropna(subset=["jaccard", "f1"])
    summary: Dict[str, Any] = {
        "count_total": float(len(results_df)),
        "count_scored": float(len(valid)),
        "missing_outputs": float(len(results_df) - len(valid)),
    }
    if valid.empty:
        summary.update(
            {
                "mean_jaccard": np.nan,
                "std_jaccard": np.nan,
                "mean_f1": np.nan,
                "std_f1": np.nan,
            }
        )
    else:
        summary.update(
            {
                "mean_jaccard": float(valid["jaccard"].mean()),
                "std_jaccard": float(valid["jaccard"].std(ddof=0)),
                "mean_f1": float(valid["f1"].mean()),
                "std_f1": float(valid["f1"].std(ddof=0)),
            }
        )
    return summary


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and np.isfinite(value)


def _safe_metric_name(value: str) -> str:
    metric_name = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return metric_name or "unknown"


def _resolve_weight_column(
    df: pd.DataFrame,
    weights_column: Optional[str],
) -> Optional[str]:
    if weights_column:
        if weights_column not in df.columns:
            raise ValueError(
                f"Requested weights column '{weights_column}' not found in QA parquet."
            )
        return weights_column

    for candidate in WEIGHT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _extract_competitor_weights(
    df: pd.DataFrame,
    competitors: Sequence[str],
    weights_column: Optional[str],
) -> Optional[Dict[str, float]]:
    if weights_column is None:
        return None

    if "competitor" not in df.columns:
        return None

    weight_df = df[["competitor", weights_column]].copy()
    weight_df[weights_column] = pd.to_numeric(
        weight_df[weights_column], errors="coerce"
    )
    weight_df = weight_df.dropna(subset=[weights_column])
    if weight_df.empty:
        return None

    per_competitor = (
        weight_df.groupby("competitor", dropna=False)[weights_column].median().to_dict()
    )
    resolved: Dict[str, float] = {}
    for competitor in competitors:
        value = per_competitor.get(competitor)
        if value is None or not _is_finite_number(value):
            return None
        resolved[competitor] = float(value)

    return resolved


def _extract_split_core_metrics(
    eval_metrics_by_split: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    core_metrics: Dict[str, float] = {}
    for split_name, split_metrics in eval_metrics_by_split.items():
        safe_split = _safe_metric_name(split_name)
        for metric_name in ("mean_jaccard", "mean_f1", "count"):
            value = split_metrics.get(metric_name)
            if value is not None and _is_finite_number(value):
                core_metrics[f"{safe_split}_{metric_name}"] = float(value)
    return core_metrics


def _select_ranking_split(split_metrics: Dict[str, float]) -> str:
    for split_name in ("test", "validation", "overall"):
        jaccard = split_metrics.get(f"{split_name}_mean_jaccard")
        f1 = split_metrics.get(f"{split_name}_mean_f1")
        if _is_finite_number(jaccard) or _is_finite_number(f1):
            return split_name
    return "overall"


@contextmanager
def fusion_job_dir(
    output_base_dir: Path,
    keep_job_dir: bool,
    explicit_job_dir: Optional[Path],
) -> Iterator[Path]:
    if explicit_job_dir is not None:
        explicit_job_dir.mkdir(parents=True, exist_ok=True)
        yield explicit_job_dir
        return

    if keep_job_dir:
        persistent_job_dir = (
            output_base_dir / f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        persistent_job_dir.mkdir(parents=True, exist_ok=True)
        yield persistent_job_dir
        return

    with tempfile.TemporaryDirectory(
        prefix="fusion_crops_job_", dir=str(output_base_dir)
    ) as temp_dir:
        yield Path(temp_dir)


def _resolve_output_path(output_dir: Union[Path, str]) -> Path:
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    return PROJECT_ROOT / output_path


def _resolve_tracking_path(tracking_path: Union[Path, str]) -> Path:
    path = Path(tracking_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _safe_run_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    token = token.strip("_")
    return token or "unknown"


def run_crops_fusion_experiment(
    qa_parquet: Union[Path, str],
    output_dir: Union[Path, str] = "fusion_results_crops",
    models: Sequence[str] = (),
    all_models: bool = False,
    flat_models_only: bool = False,
    num_threads: int = 4,
    mlflow_experiment: str = "fusion-crops-baseline",
    mlflow_tracking_path: Union[Path, str] = MLFLOW_TRACKING_PATH,
    weights_column: Optional[str] = None,
    skip_fusion: bool = False,
    keep_job_dir: bool = False,
    job_dir: Optional[Union[Path, str]] = None,
    chunk_size: int = 0,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run fusion on QA crops for multiple models with MLflow tracking."""
    qa_parquet_path = Path(qa_parquet).expanduser().resolve()
    if not qa_parquet_path.exists():
        raise FileNotFoundError(f"QA parquet not found: {qa_parquet_path}")

    selected_models_requested = select_models(
        models=models,
        all_models=all_models,
        flat_models_only=flat_models_only,
    )

    output_base_dir = _resolve_output_path(output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = _resolve_tracking_path(mlflow_tracking_path)
    tracking_uri.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(mlflow_experiment)

    if not DEFAULT_JAR_PATH.exists():
        raise FileNotFoundError(f"Fusion JAR not found: {DEFAULT_JAR_PATH}")

    df = pd.read_parquet(qa_parquet_path)
    validate_columns(df)
    cell_groups, cell_splits, cell_metadata = build_cell_groups(df, qa_parquet_path)
    if not cell_groups:
        raise ValueError("No valid cells found in QA parquet after path resolution")
    if "split" not in df.columns:
        logger.warning(
            "QA parquet has no 'split' column. Split-specific evaluation metrics will be unavailable."
        )
    dataset_tag = infer_dataset_name_from_text(
        [
            qa_parquet_path,
            *(
                df["gt_image"].dropna().astype(str).head(10).tolist()
                if "gt_image" in df.columns
                else []
            ),
        ]
    )
    split_tag = infer_split_from_dataframe(df)

    competitors = sorted(str(value) for value in df["competitor"].dropna().unique())
    if not competitors:
        raise ValueError("No competitors found in QA parquet")
    competitor_dir_names = build_competitor_dir_names(competitors)
    resolved_weights_column = _resolve_weight_column(df, weights_column)
    competitor_weights = _extract_competitor_weights(
        df=df,
        competitors=competitors,
        weights_column=resolved_weights_column,
    )

    user_weighted_models_requested = [
        model for model in selected_models_requested if model in USER_WEIGHTED_MODELS
    ]
    skipped_weighted_models: List[str] = []
    selected_models = list(selected_models_requested)
    if user_weighted_models_requested and competitor_weights is None:
        skipped_weighted_models = user_weighted_models_requested
        selected_models = [
            model
            for model in selected_models_requested
            if model not in USER_WEIGHTED_MODELS
        ]
        if resolved_weights_column is None:
            logger.warning(
                "Skipping user-weighted fusion models (no weights column provided/found): %s",
                ", ".join(skipped_weighted_models),
            )
        else:
            logger.warning(
                "Skipping user-weighted fusion models (weights missing/invalid in column '%s'): %s",
                resolved_weights_column,
                ", ".join(skipped_weighted_models),
            )

    if not selected_models:
        raise ValueError(
            "No runnable models left after filtering. Provide non-weighted models or valid competitor weights."
        )

    explicit_job_path = None
    if job_dir is not None:
        explicit_job_path = Path(job_dir).expanduser().resolve()

    logger.info("=" * 70)
    logger.info("Fusion Crops Experiment")
    logger.info("  QA parquet: %s", qa_parquet_path)
    logger.info("  Requested models: %s", selected_models_requested)
    logger.info("  Models to run: %s", selected_models)
    logger.info("  Cells: %d", len(cell_groups))
    logger.info("  Competitors: %s", competitors)
    logger.info("  Weights column: %s", resolved_weights_column or "<none>")
    logger.info("  Weights available: %s", competitor_weights is not None)
    if skipped_weighted_models:
        logger.info("  Skipped weighted models: %s", skipped_weighted_models)
    if any(competitor_dir_names[c] != c for c in competitors):
        logger.info("  Competitor dir names: %s", competitor_dir_names)
    logger.info("  Output dir: %s", output_base_dir)
    logger.info("=" * 70)

    run_name = (
        f"{_safe_run_token(dataset_tag)}_{_safe_run_token(split_tag)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    model_summaries: List[Dict[str, Any]] = []

    with fusion_job_dir(
        output_base_dir=output_base_dir,
        keep_job_dir=keep_job_dir,
        explicit_job_dir=explicit_job_path,
    ) as active_job_dir:
        logger.info("Using fusion job directory: %s", active_job_dir)
        mapping = prepare_fusion_job(
            active_job_dir / "job",
            cell_groups,
            competitors,
            cell_metadata=cell_metadata,
            competitor_dir_names=competitor_dir_names,
        )
        if not mapping:
            raise ValueError("No cells were staged for fusion; nothing to process")

        unweighted_job_file = write_job_file(
            active_job_dir / "job",
            competitors,
            weighted=False,
            competitor_dir_names=competitor_dir_names,
        )
        weighted_job_file: Optional[Path] = None
        if any(model in JOB_WEIGHT_FORMAT_MODELS for model in selected_models):
            effective_job_weights = (
                competitor_weights
                if competitor_weights is not None
                else {competitor: 1.0 for competitor in competitors}
            )
            weighted_job_file = write_job_file(
                active_job_dir / "job",
                competitors,
                weighted=True,
                competitor_dir_names=competitor_dir_names,
                competitor_weights=effective_job_weights,
            )

        persistent_gt_dir = output_base_dir / "_evaluation_gt"
        _persist_gt_masks(active_job_dir / "job" / "GT", persistent_gt_dir, mapping)
        gt_lookup = _build_key_path_lookup(mapping, persistent_gt_dir, "mask")

        all_indices = list(mapping.keys())
        if chunk_size > 0:
            chunks = chunk_indices(all_indices, chunk_size)
            chunking_enabled = True
        else:
            chunks = [all_indices]
            chunking_enabled = False

        with mlflow.start_run(run_name=run_name) as parent_run:
            set_common_mlflow_tags(
                dataset=dataset_tag,
                split=split_tag,
                repo_root=PROJECT_ROOT,
            )
            mlflow.set_tag("run_kind", "experiment_parent")
            mlflow.set_tag("parent_scope", "dataset_split")
            mlflow.log_params(
                {
                    "qa_parquet": str(qa_parquet_path),
                    "models_requested": ",".join(selected_models_requested),
                    "models_run": ",".join(selected_models),
                    "num_threads": num_threads,
                    "chunk_size_requested": chunk_size,
                    "chunking_enabled": chunking_enabled,
                    "num_chunks": len(chunks),
                    "cells": len(mapping),
                    "competitors": ",".join(competitors),
                    "weights_column": resolved_weights_column or "",
                    "weights_available": competitor_weights is not None,
                    "weighted_models_skipped": ",".join(skipped_weighted_models),
                    "skip_fusion": skip_fusion,
                }
            )

            all_model_summaries: List[Dict[str, Any]] = []
            for model in selected_models:
                model_lower = model.lower()
                model_output_dir = output_base_dir / model_lower
                model_output_dir.mkdir(parents=True, exist_ok=True)

                with mlflow.start_run(run_name=model, nested=True) as model_run:
                    set_common_mlflow_tags(
                        dataset=dataset_tag,
                        split=split_tag,
                        repo_root=PROJECT_ROOT,
                    )
                    needs_user_weights = model in USER_WEIGHTED_MODELS
                    needs_weighted_job_format = model in JOB_WEIGHT_FORMAT_MODELS
                    mlflow.log_params(
                        {
                            "model": model,
                            "needs_user_weights": needs_user_weights,
                            "needs_weighted_job_format": needs_weighted_job_format,
                            "num_threads": num_threads,
                            "chunk_size_requested": chunk_size,
                            "chunking_enabled": chunking_enabled,
                            "num_chunks": len(chunks),
                            "cells": len(mapping),
                            "competitors_count": len(competitors),
                            "weights_column": resolved_weights_column or "",
                            "weights_available": competitor_weights is not None,
                            "skip_fusion": skip_fusion,
                        }
                    )
                    mlflow.set_tag("run_kind", "model_run")
                    mlflow.set_tag("fusion_status", "started")

                    output_pattern = str(model_output_dir / "fused_TTTT.tif")
                    fusion_success = True
                    fusion_error = ""
                    chunk_fallback_used = False

                    if not skip_fusion:
                        if needs_weighted_job_format:
                            if weighted_job_file is None:
                                raise ValueError(
                                    f"Missing weighted job file for model '{model}'."
                                )
                            selected_job_file = weighted_job_file
                        else:
                            selected_job_file = unweighted_job_file

                        def _run_chunk_sequence(
                            chunk_sequence: List[List[int]],
                        ) -> Tuple[bool, str]:
                            for chunk_index, chunk in enumerate(chunk_sequence):
                                time_points = ",".join(str(value) for value in chunk)
                                logger.info(
                                    "Model %s: chunk %d/%d (%d cells)",
                                    model,
                                    chunk_index + 1,
                                    len(chunk_sequence),
                                    len(chunk),
                                )
                                try:
                                    fuse_segmentations(
                                        jar_path=str(DEFAULT_JAR_PATH),
                                        job_file_path=str(selected_job_file),
                                        output_path_pattern=output_pattern,
                                        time_points=time_points,
                                        num_threads=num_threads,
                                        fusion_model=FusionModel[model],
                                        threshold=DEFAULT_FUSION_THRESHOLD,
                                        debug=debug,
                                    )
                                except Exception as exc:
                                    return False, str(exc)
                            return True, ""

                        fusion_success, fusion_error = _run_chunk_sequence(chunks)

                        # Best-effort fallback: retry with smaller chunks if single-call mode fails.
                        if (
                            not fusion_success
                            and not chunking_enabled
                            and len(all_indices) > 1
                        ):
                            fallback_chunk_size = min(200, len(all_indices))
                            fallback_chunks = chunk_indices(
                                all_indices, fallback_chunk_size
                            )
                            if len(fallback_chunks) > 1:
                                chunk_fallback_used = True
                                logger.warning(
                                    "Single-call fusion failed for model=%s. "
                                    "Retrying with chunk_size=%d (%d chunks).",
                                    model,
                                    fallback_chunk_size,
                                    len(fallback_chunks),
                                )
                                fusion_success, fusion_error = _run_chunk_sequence(
                                    fallback_chunks
                                )

                        if not fusion_success:
                            logger.error(
                                "Fusion failed for model=%s: %s",
                                model,
                                fusion_error,
                            )

                    model_results_df = evaluate_model_results(
                        fusion_out_dir=model_output_dir,
                        gt_dir=persistent_gt_dir,
                        mapping=mapping,
                        cell_splits=cell_splits,
                        model=model,
                    )
                    model_results_df["fusion_success"] = fusion_success
                    model_results_df["fusion_error"] = fusion_error
                    model_results_df["chunk_fallback_used"] = float(chunk_fallback_used)

                    fused_output_diagnostics = inspect_fused_outputs(
                        model_output_dir, mapping
                    )
                    for name, value in fused_output_diagnostics.items():
                        model_results_df[name] = value

                    results_path = model_output_dir / "results.csv"
                    model_results_df.to_csv(results_path, index=False)
                    mlflow.log_artifact(str(results_path))

                    model_summary = summarize_results(model_results_df)
                    model_summary.update(fused_output_diagnostics)

                    fused_lookup = _build_key_path_lookup(
                        mapping, model_output_dir, "fused_"
                    )
                    model_parquet = (
                        model_output_dir
                        / f"{qa_parquet_path.stem}_{model_lower}_with_fused.parquet"
                    )
                    eval_parquet = (
                        model_output_dir
                        / f"{qa_parquet_path.stem}_{model_lower}_cell_eval.parquet"
                    )
                    eval_metrics_by_split: Dict[str, Dict[str, float]] = {}
                    try:
                        qa_with_paths_df = _enrich_qa_with_paths(
                            qa_df=df,
                            fused_lookup=fused_lookup,
                            gt_lookup=gt_lookup,
                            fused_column=model_lower,
                        )
                        qa_with_paths_df.to_parquet(model_parquet, index=False)
                        mlflow.log_artifact(str(model_parquet))

                        cell_eval_df = _build_cell_level_eval_df(
                            qa_with_paths_df, model_lower
                        )
                        cell_eval_df.to_parquet(eval_parquet, index=False)
                        mlflow.log_artifact(str(eval_parquet))

                        eval_metrics_by_split = evaluate_by_split(
                            eval_parquet,
                            model_lower,
                            gt_column="crop_gt_path",
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed parquet mapping/evaluation for model=%s: %s",
                            model,
                            exc,
                        )
                        model_summary["parquet_eval_error"] = 1.0

                    split_core_metrics = _extract_split_core_metrics(
                        eval_metrics_by_split
                    )
                    overall_jaccard = split_core_metrics.get("overall_mean_jaccard")
                    overall_f1 = split_core_metrics.get("overall_mean_f1")
                    overall_count = split_core_metrics.get("overall_count")
                    if overall_jaccard is not None and _is_finite_number(
                        overall_jaccard
                    ):
                        model_summary["mean_jaccard"] = float(overall_jaccard)
                    if overall_f1 is not None and _is_finite_number(overall_f1):
                        model_summary["mean_f1"] = float(overall_f1)
                    if overall_count is not None and _is_finite_number(overall_count):
                        model_summary["count_scored"] = float(overall_count)

                    for metric_name, metric_value in split_core_metrics.items():
                        model_summary[f"eval_{metric_name}"] = float(metric_value)

                    ranking_split = _select_ranking_split(split_core_metrics)
                    ranking_mean_jaccard = split_core_metrics.get(
                        f"{ranking_split}_mean_jaccard", np.nan
                    )
                    ranking_mean_f1 = split_core_metrics.get(
                        f"{ranking_split}_mean_f1", np.nan
                    )

                    model_summary["model"] = model
                    model_summary["fusion_success"] = float(fusion_success)
                    model_summary["chunk_fallback_used"] = float(chunk_fallback_used)
                    model_summary["selection_split"] = ranking_split
                    model_summary["selection_mean_jaccard"] = (
                        float(ranking_mean_jaccard)
                        if _is_finite_number(ranking_mean_jaccard)
                        else np.nan
                    )
                    model_summary["selection_mean_f1"] = (
                        float(ranking_mean_f1)
                        if _is_finite_number(ranking_mean_f1)
                        else np.nan
                    )
                    model_summary["mlflow_run_id"] = model_run.info.run_id
                    all_model_summaries.append(model_summary.copy())

                    model_results_path = model_output_dir / f"{model_lower}_results.csv"
                    model_results_df.to_csv(model_results_path, index=False)
                    model_summary_path = model_output_dir / f"{model_lower}_summary.csv"
                    pd.DataFrame([model_summary]).to_csv(
                        model_summary_path, index=False
                    )
                    mlflow.log_artifact(str(model_results_path))
                    mlflow.log_artifact(str(model_summary_path))

                    mlflow_metrics: Dict[str, float] = {}
                    for split_name in ("train", "validation", "test", "overall"):
                        for metric_name in ("mean_jaccard", "mean_f1", "count"):
                            key = f"{split_name}_{metric_name}"
                            split_value = split_core_metrics.get(key)
                            if split_value is not None and _is_finite_number(
                                split_value
                            ):
                                mlflow_metrics[key] = float(split_value)

                    diagnostics_metric_map = {
                        "outputs_expected": fused_output_diagnostics.get(
                            "expected_outputs"
                        ),
                        "outputs_present": fused_output_diagnostics.get(
                            "present_outputs"
                        ),
                        "outputs_missing": fused_output_diagnostics.get(
                            "missing_fused_outputs"
                        ),
                        "outputs_nonempty": fused_output_diagnostics.get(
                            "nonempty_fused_outputs"
                        ),
                    }
                    for (
                        metric_name,
                        diag_metric_value,
                    ) in diagnostics_metric_map.items():
                        if diag_metric_value is not None and _is_finite_number(
                            diag_metric_value
                        ):
                            mlflow_metrics[metric_name] = float(diag_metric_value)

                    if _is_finite_number(ranking_mean_jaccard):
                        mlflow_metrics["selection_mean_jaccard"] = float(
                            ranking_mean_jaccard
                        )
                    if _is_finite_number(ranking_mean_f1):
                        mlflow_metrics["selection_mean_f1"] = float(ranking_mean_f1)
                    if mlflow_metrics:
                        mlflow.log_metrics(mlflow_metrics)

                    if fusion_success:
                        mlflow.set_tag("fusion_status", "success")
                    else:
                        mlflow.set_tag("fusion_status", "failed")
                        if fusion_error:
                            mlflow.set_tag("fusion_error", fusion_error[:250])
                    mlflow.set_tag("selection_split", ranking_split)

                    model_summaries.append(
                        {
                            "model": model,
                            "status": (
                                "success"
                                if _is_finite_number(
                                    model_summary["selection_mean_jaccard"]
                                )
                                else "failed"
                            ),
                            "best_mean_jaccard": model_summary[
                                "selection_mean_jaccard"
                            ],
                            "best_mean_f1": model_summary["selection_mean_f1"],
                        }
                    )

            summary_df = pd.DataFrame(model_summaries)
            summary_path = output_base_dir / "fusion_crops_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            mlflow.log_artifact(str(summary_path))

            leaderboard_df = pd.DataFrame(all_model_summaries)
            leaderboard_path = output_base_dir / "fusion_crops_leaderboard.csv"
            if leaderboard_df.empty:
                leaderboard_df = pd.DataFrame(
                    columns=[
                        "model",
                        "selection_split",
                        "selection_mean_jaccard",
                        "selection_mean_f1",
                        "eval_test_mean_jaccard",
                        "eval_test_mean_f1",
                        "eval_overall_mean_jaccard",
                        "eval_overall_mean_f1",
                        "mlflow_run_id",
                    ]
                )
            else:
                required_columns = [
                    "eval_test_mean_jaccard",
                    "eval_test_mean_f1",
                    "selection_mean_jaccard",
                    "selection_mean_f1",
                    "eval_overall_mean_jaccard",
                    "eval_overall_mean_f1",
                    "fusion_success",
                    "selection_split",
                    "mlflow_run_id",
                ]
                for column_name in required_columns:
                    if column_name not in leaderboard_df.columns:
                        leaderboard_df[column_name] = np.nan

                sort_test_jaccard = leaderboard_df["eval_test_mean_jaccard"].fillna(
                    -1.0
                )
                sort_test_f1 = leaderboard_df["eval_test_mean_f1"].fillna(-1.0)
                sort_selection_jaccard = leaderboard_df[
                    "selection_mean_jaccard"
                ].fillna(-1.0)
                leaderboard_df = leaderboard_df.assign(
                    _sort_test_jaccard=sort_test_jaccard,
                    _sort_test_f1=sort_test_f1,
                    _sort_selection_jaccard=sort_selection_jaccard,
                ).sort_values(
                    by=[
                        "_sort_test_jaccard",
                        "_sort_test_f1",
                        "_sort_selection_jaccard",
                    ],
                    ascending=False,
                )
                leaderboard_df = leaderboard_df.drop(
                    columns=[
                        "_sort_test_jaccard",
                        "_sort_test_f1",
                        "_sort_selection_jaccard",
                    ]
                ).reset_index(drop=True)
            leaderboard_df.to_csv(leaderboard_path, index=False)
            mlflow.log_artifact(str(leaderboard_path))

            successful_models = int((summary_df["status"] == "success").sum())
            successful_model_runs = (
                int((leaderboard_df["fusion_success"] == 1.0).sum())
                if "fusion_success" in leaderboard_df.columns
                else 0
            )
            mlflow.log_metrics(
                {
                    "models_successful": successful_models,
                    "models_failed": int(len(summary_df) - successful_models),
                    "model_runs_total": int(len(leaderboard_df)),
                    "model_runs_successful": successful_model_runs,
                }
            )
            if not leaderboard_df.empty:
                best_test_iou_rows = leaderboard_df.dropna(
                    subset=["eval_test_mean_jaccard"]
                )
                if not best_test_iou_rows.empty:
                    best_test_iou = best_test_iou_rows.iloc[0]
                    mlflow.set_tag("best_test_iou_model", str(best_test_iou["model"]))
                    if _is_finite_number(best_test_iou["eval_test_mean_jaccard"]):
                        mlflow.log_metric(
                            "best_test_mean_jaccard",
                            float(best_test_iou["eval_test_mean_jaccard"]),
                        )

                best_test_f1_rows = leaderboard_df.dropna(
                    subset=["eval_test_mean_f1"]
                ).sort_values("eval_test_mean_f1", ascending=False)
                if not best_test_f1_rows.empty:
                    best_test_f1 = best_test_f1_rows.iloc[0]
                    mlflow.set_tag("best_test_f1_model", str(best_test_f1["model"]))
                    if _is_finite_number(best_test_f1["eval_test_mean_f1"]):
                        mlflow.log_metric(
                            "best_test_mean_f1",
                            float(best_test_f1["eval_test_mean_f1"]),
                        )

            persisted_job_dir = ""
            if keep_job_dir or explicit_job_path is not None:
                persisted_job_dir = str(active_job_dir)

            return {
                "mlflow_run_id": parent_run.info.run_id,
                "summary_path": str(summary_path),
                "leaderboard_path": str(leaderboard_path),
                "output_dir": str(output_base_dir),
                "job_dir": persisted_job_dir,
                "models_run": selected_models,
                "weighted_models_skipped": skipped_weighted_models,
            }
