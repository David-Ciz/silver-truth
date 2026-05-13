from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import tifffile

from silver_truth.data_processing.compression import write_tiff_lossless
from silver_truth.metrics.metrics import calculate_labelwise_scores

logger = logging.getLogger(__name__)

RECONSTRUCTION_COLUMNS = [
    "gt_image",
    "recon_crop_y_start",
    "recon_crop_y_end",
    "recon_crop_x_start",
    "recon_crop_x_end",
]


def has_reconstruction_metadata(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in RECONSTRUCTION_COLUMNS)


def _to_numpy(array_like: Any) -> np.ndarray:
    value = array_like
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _to_binary_crop(array_like: Any, threshold: float) -> np.ndarray:
    array = _to_numpy(array_like)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D crop mask, got shape={array.shape}")
    return (array > threshold).astype(np.uint8) * 255


def _to_score_crop(array_like: Any) -> np.ndarray:
    array = _to_numpy(array_like)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D crop mask, got shape={array.shape}")
    return array.astype(np.float32, copy=False)


def _paste_crop(
    canvas: np.ndarray, crop: np.ndarray, y_start: int, x_start: int
) -> None:
    y0 = max(0, y_start)
    x0 = max(0, x_start)
    y1 = min(canvas.shape[0], y_start + crop.shape[0])
    x1 = min(canvas.shape[1], x_start + crop.shape[1])
    if y1 <= y0 or x1 <= x0:
        return

    src_y0 = max(0, -y_start)
    src_x0 = max(0, -x_start)
    src_y1 = src_y0 + (y1 - y0)
    src_x1 = src_x0 + (x1 - x0)

    canvas[y0:y1, x0:x1] = np.maximum(
        canvas[y0:y1, x0:x1], crop[src_y0:src_y1, src_x0:src_x1]
    )


def _compute_iou_f1(segmentation: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    seg_bin = segmentation > 0
    gt_bin = gt > 0

    intersection = np.logical_and(seg_bin, gt_bin).sum()
    union = np.logical_or(seg_bin, gt_bin).sum()
    iou = float(intersection / union) if union > 0 else 1.0

    tp = intersection
    fp = np.logical_and(seg_bin, ~gt_bin).sum()
    fn = np.logical_and(~seg_bin, gt_bin).sum()
    f1_denominator = (2 * tp) + fp + fn
    f1 = float((2 * tp) / f1_denominator) if f1_denominator > 0 else 1.0
    return iou, f1


def _coerce_positive_label(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        numeric = int(value.strip())
        return numeric if numeric > 0 else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not float(numeric).is_integer():
        return None
    as_int = int(numeric)
    return as_int if as_int > 0 else None


def _resolve_priority_columns(
    df: pd.DataFrame, priority_columns: Sequence[str] | None
) -> list[str]:
    requested = list(priority_columns) if priority_columns is not None else []
    return [column for column in requested if column in df.columns]


def _row_priority(row: pd.Series, priority_columns: Sequence[str]) -> float:
    for column in priority_columns:
        value = row.get(column)
        if pd.notna(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _group_sort_columns(
    group_df: pd.DataFrame, priority_columns: Sequence[str]
) -> pd.DataFrame:
    enriched = group_df.copy()
    enriched["_label_sort_key"] = enriched["label"].map(lambda value: str(value))
    if not priority_columns:
        return enriched.sort_values(["_label_sort_key"], kind="stable").reset_index(
            drop=True
        )
    enriched["_reconstruction_priority"] = enriched.apply(
        lambda row: _row_priority(row, priority_columns), axis=1
    )
    return enriched.sort_values(
        ["_reconstruction_priority", "_label_sort_key"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def _paste_labeled_crop(
    *,
    label_canvas: np.ndarray,
    score_canvas: np.ndarray,
    priority_canvas: np.ndarray,
    crop_scores: np.ndarray,
    label_value: int,
    threshold: float,
    y_start: int,
    x_start: int,
    row_priority: float,
) -> bool:
    y0 = max(0, y_start)
    x0 = max(0, x_start)
    y1 = min(label_canvas.shape[0], y_start + crop_scores.shape[0])
    x1 = min(label_canvas.shape[1], x_start + crop_scores.shape[1])
    if y1 <= y0 or x1 <= x0:
        return False

    src_y0 = max(0, -y_start)
    src_x0 = max(0, -x_start)
    src_y1 = src_y0 + (y1 - y0)
    src_x1 = src_x0 + (x1 - x0)

    crop_region = crop_scores[src_y0:src_y1, src_x0:src_x1]
    active_mask = crop_region > threshold
    if not np.any(active_mask):
        return False

    canvas_priority = priority_canvas[y0:y1, x0:x1]
    canvas_scores = score_canvas[y0:y1, x0:x1]
    should_update = active_mask & (
        (row_priority > canvas_priority)
        | ((row_priority == canvas_priority) & (crop_region > canvas_scores))
    )
    if not np.any(should_update):
        return False

    label_canvas_view = label_canvas[y0:y1, x0:x1]
    label_canvas_view[should_update] = label_value
    canvas_priority[should_update] = row_priority
    canvas_scores[should_update] = crop_region[should_update]
    return True


def _empty_reconstruction_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "campaign_number",
            "original_image_key",
            "gt_image",
            "reconstructed_path",
            "split",
            "cells_considered",
            "cells_placed",
            "labels_scored",
            "iou",
            "f1",
        ]
    )


def reconstruct_full_images_from_arrays(
    databank_df: pd.DataFrame,
    predicted_crops: Sequence[Any],
    output_dir: Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Reconstruct full-image binary segmentations by placing predicted per-cell crops
    back into image coordinates and evaluate against full GT masks.
    """
    if len(databank_df) != len(predicted_crops):
        raise ValueError(
            "Length mismatch between databank rows and predicted crops: "
            f"{len(databank_df)} vs {len(predicted_crops)}."
        )
    if not has_reconstruction_metadata(databank_df):
        missing = [c for c in RECONSTRUCTION_COLUMNS if c not in databank_df.columns]
        raise ValueError(
            "Databank is missing reconstruction metadata columns: " + ", ".join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = databank_df.reset_index(drop=True).copy()
    rows["_prediction_index"] = np.arange(len(rows))

    result_rows = []
    for gt_image, group_df in rows.groupby("gt_image", dropna=False):
        if pd.isna(gt_image):
            logger.warning("Skipping reconstruction group with missing gt_image.")
            continue

        gt_path = Path(str(gt_image))
        if not gt_path.exists():
            logger.warning("GT image not found for reconstruction: %s", gt_path)
            continue

        gt_full = tifffile.imread(gt_path)
        if gt_full.ndim > 2:
            gt_full = np.squeeze(gt_full)
        if gt_full.ndim != 2:
            logger.warning(
                "Skipping GT with unsupported shape %s: %s", gt_full.shape, gt_path
            )
            continue

        reconstructed = np.zeros(gt_full.shape, dtype=np.uint8)
        placed_cells = 0
        for _, row in group_df.iterrows():
            pred_idx = int(row["_prediction_index"])
            pred_crop = _to_binary_crop(predicted_crops[pred_idx], threshold)
            y_start = int(row["recon_crop_y_start"])
            x_start = int(row["recon_crop_x_start"])
            _paste_crop(reconstructed, pred_crop, y_start=y_start, x_start=x_start)
            placed_cells += 1

        first_row = group_df.iloc[0]
        campaign = str(
            first_row.get(
                "campaign_number", first_row.get("campaign", "unknown_campaign")
            )
        )
        image_key = str(
            first_row.get(
                "original_image_key", first_row.get("image_id", "unknown_image")
            )
        )
        output_path = output_dir / f"{campaign}_{image_key}_reconstructed.tif"
        write_tiff_lossless(output_path, reconstructed)

        iou, f1 = _compute_iou_f1(reconstructed, gt_full)
        result_rows.append(
            {
                "campaign_number": campaign,
                "original_image_key": image_key,
                "gt_image": str(gt_path),
                "reconstructed_path": str(output_path),
                "split": first_row.get("split", None),
                "cells_placed": int(placed_cells),
                "iou": iou,
                "f1": f1,
            }
        )

    return pd.DataFrame(result_rows)


def _ensure_recon_crop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise crop-coordinate column names so that both naming conventions are accepted.

    ``run-crops-experiment`` writes ``crop_y_start / crop_y_end / crop_x_start /
    crop_x_end`` (plain QA-parquet names).  ``reconstruct_full_images_from_arrays``
    expects the ``recon_crop_*`` prefix used by the ensemble databank builder.

    This function returns a *copy* of the dataframe with the ``recon_crop_*`` columns
    added when only the plain ``crop_*`` variants are present.  The original frame is
    never mutated.
    """
    _PLAIN = ["crop_y_start", "crop_y_end", "crop_x_start", "crop_x_end"]
    _RECON = [
        "recon_crop_y_start",
        "recon_crop_y_end",
        "recon_crop_x_start",
        "recon_crop_x_end",
    ]

    if all(c in df.columns for c in _RECON):
        return df  # already correct — nothing to do

    if all(c in df.columns for c in _PLAIN):
        df = df.copy()
        for plain, recon in zip(_PLAIN, _RECON):
            df[recon] = df[plain]
        logger.debug(
            "Aliased plain crop columns (%s) → recon_crop_* columns for reconstruction.",
            _PLAIN,
        )
        return df

    # Neither set is complete; let the downstream validator raise a clear error.
    return df


def reconstruct_full_images_from_paths(
    databank_df: pd.DataFrame,
    fused_path_column: str,
    output_dir: Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Reconstruct full images from on-disk fused per-cell masks referenced by a column.

    Accepts dataframes that use either the ``recon_crop_*`` column naming (ensemble
    databank) or the plain ``crop_*`` naming produced by ``run-crops-experiment``.
    Both conventions are normalised before reconstruction is attempted.
    """
    if fused_path_column not in databank_df.columns:
        raise ValueError(
            f"Missing fused path column '{fused_path_column}' in databank dataframe."
        )
    databank_df = _ensure_recon_crop_columns(databank_df)

    predicted_crops = []
    valid_indices = []
    for index, row in databank_df.iterrows():
        fused_path = row[fused_path_column]
        if pd.isna(fused_path):
            continue
        mask_path = Path(str(fused_path))
        if not mask_path.exists():
            logger.warning("Fused mask not found: %s", mask_path)
            continue
        img = tifffile.imread(mask_path)
        # Stacked QA crops are multi-channel (C, H, W); channel 1 is the
        # competitor segmentation mask.  Single-channel fused outputs are
        # used as-is.
        if img.ndim == 3:
            img = img[1]
        predicted_crops.append(img)
        valid_indices.append(index)

    if not valid_indices:
        return pd.DataFrame(
            columns=[
                "campaign_number",
                "original_image_key",
                "gt_image",
                "reconstructed_path",
                "split",
                "cells_placed",
                "iou",
                "f1",
            ]
        )

    valid_df = databank_df.loc[valid_indices].reset_index(drop=True)
    return reconstruct_full_images_from_arrays(
        databank_df=valid_df,
        predicted_crops=predicted_crops,
        output_dir=output_dir,
        threshold=threshold,
    )


def reconstruct_labeled_full_images_from_arrays(
    databank_df: pd.DataFrame,
    predicted_crops: Sequence[Any],
    output_dir: Path,
    threshold: float = 0.5,
    priority_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Reconstruct labeled full-image segmentations and score them with the canonical
    per-label full-image metric used for competitor/silver-truth baselines.

    Each row must identify a logical cell via its ``label`` and reconstruction
    coordinates. Predicted positive pixels are painted back into image space using
    that label value. Overlapping cells are resolved deterministically:

    1. Higher row-level priority wins when ``priority_columns`` are provided.
    2. Within equal-priority rows, higher per-pixel score wins.
    3. Exact ties preserve the first-written label (stable ordering).
    """
    if len(databank_df) != len(predicted_crops):
        raise ValueError(
            "Length mismatch between databank rows and predicted crops: "
            f"{len(databank_df)} vs {len(predicted_crops)}."
        )
    databank_df = _ensure_recon_crop_columns(databank_df)
    if not has_reconstruction_metadata(databank_df):
        missing = [c for c in RECONSTRUCTION_COLUMNS if c not in databank_df.columns]
        raise ValueError(
            "Databank is missing reconstruction metadata columns: " + ", ".join(missing)
        )
    if "label" not in databank_df.columns:
        raise ValueError(
            "Databank must include a 'label' column for labeled reconstruction."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = databank_df.reset_index(drop=True).copy()
    rows["_prediction_index"] = np.arange(len(rows))
    active_priority_columns = _resolve_priority_columns(rows, priority_columns)

    result_rows = []
    for gt_image, group_df in rows.groupby("gt_image", dropna=False):
        if pd.isna(gt_image):
            logger.warning("Skipping reconstruction group with missing gt_image.")
            continue

        gt_path = Path(str(gt_image))
        if not gt_path.exists():
            logger.warning("GT image not found for labeled reconstruction: %s", gt_path)
            continue

        gt_full = tifffile.imread(gt_path)
        if gt_full.ndim > 2:
            gt_full = np.squeeze(gt_full)
        if gt_full.ndim != 2:
            logger.warning(
                "Skipping GT with unsupported shape %s: %s", gt_full.shape, gt_path
            )
            continue

        reconstructed = np.zeros(gt_full.shape, dtype=gt_full.dtype)
        pixel_scores = np.full(gt_full.shape, -np.inf, dtype=np.float32)
        pixel_priorities = np.full(gt_full.shape, -np.inf, dtype=np.float32)

        ordered_group = _group_sort_columns(group_df, active_priority_columns)
        cells_considered = int(len(ordered_group))
        cells_placed = 0
        for _, row in ordered_group.iterrows():
            pred_idx = int(row["_prediction_index"])
            label_value = _coerce_positive_label(row.get("label"))
            if label_value is None:
                logger.warning(
                    "Skipping reconstruction row with invalid label=%r for gt=%s",
                    row.get("label"),
                    gt_path,
                )
                continue

            crop_scores = _to_score_crop(predicted_crops[pred_idx])
            row_priority = _row_priority(row, active_priority_columns)
            pasted = _paste_labeled_crop(
                label_canvas=reconstructed,
                score_canvas=pixel_scores,
                priority_canvas=pixel_priorities,
                crop_scores=crop_scores,
                label_value=label_value,
                threshold=threshold,
                y_start=int(row["recon_crop_y_start"]),
                x_start=int(row["recon_crop_x_start"]),
                row_priority=row_priority,
            )
            if pasted:
                cells_placed += 1

        first_row = ordered_group.iloc[0]
        campaign = str(
            first_row.get(
                "campaign_number", first_row.get("campaign", "unknown_campaign")
            )
        )
        image_key = str(
            first_row.get(
                "original_image_key", first_row.get("image_id", "unknown_image")
            )
        )
        output_path = output_dir / f"{campaign}_{image_key}_reconstructed.tif"
        write_tiff_lossless(output_path, reconstructed)

        labelwise_scores = calculate_labelwise_scores(gt_full, reconstructed)
        if labelwise_scores:
            mean_iou = float(
                np.mean([score["jaccard"] for score in labelwise_scores.values()])
            )
            mean_f1 = float(
                np.mean([score["f1"] for score in labelwise_scores.values()])
            )
        else:
            mean_iou = float("nan")
            mean_f1 = float("nan")

        result_rows.append(
            {
                "campaign_number": campaign,
                "original_image_key": image_key,
                "gt_image": str(gt_path),
                "reconstructed_path": str(output_path),
                "split": first_row.get("split", None),
                "cells_considered": cells_considered,
                "cells_placed": cells_placed,
                "labels_scored": int(len(labelwise_scores)),
                "iou": mean_iou,
                "f1": mean_f1,
            }
        )

    if not result_rows:
        return _empty_reconstruction_df()
    return pd.DataFrame(result_rows)


def reconstruct_labeled_full_images_from_paths(
    databank_df: pd.DataFrame,
    fused_path_column: str,
    output_dir: Path,
    threshold: float = 0.5,
    priority_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Reconstruct labeled full images from on-disk per-cell mask paths and score them
    with the same per-label metric used for competitor and silver-truth baselines.
    """
    if fused_path_column not in databank_df.columns:
        raise ValueError(
            f"Missing fused path column '{fused_path_column}' in databank dataframe."
        )
    databank_df = _ensure_recon_crop_columns(databank_df)

    predicted_crops = []
    valid_indices = []
    for index, row in databank_df.iterrows():
        fused_path = row[fused_path_column]
        if pd.isna(fused_path):
            continue
        mask_path = Path(str(fused_path))
        if not mask_path.exists():
            logger.warning("Fused mask not found: %s", mask_path)
            continue
        img = tifffile.imread(mask_path)
        # Stacked QA crops are multi-channel (C, H, W); channel 1 is the
        # competitor segmentation mask. Single-channel fused outputs are used as-is.
        if img.ndim == 3:
            img = img[1]
        predicted_crops.append(img)
        valid_indices.append(index)

    if not valid_indices:
        return _empty_reconstruction_df()

    valid_df = databank_df.loc[valid_indices].reset_index(drop=True)
    return reconstruct_labeled_full_images_from_arrays(
        databank_df=valid_df,
        predicted_crops=predicted_crops,
        output_dir=output_dir,
        threshold=threshold,
        priority_columns=priority_columns,
    )
