from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import tifffile

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
        tifffile.imwrite(output_path, reconstructed)

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


def reconstruct_full_images_from_paths(
    databank_df: pd.DataFrame,
    fused_path_column: str,
    output_dir: Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Reconstruct full images from on-disk fused per-cell masks referenced by a column.
    """
    if fused_path_column not in databank_df.columns:
        raise ValueError(
            f"Missing fused path column '{fused_path_column}' in databank dataframe."
        )

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
        predicted_crops.append(tifffile.imread(mask_path))
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
