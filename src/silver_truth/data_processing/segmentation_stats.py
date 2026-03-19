from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import click
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import find_objects

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)


def collect_segmentation_object_stats(
    segmentation_dirs: Sequence[Path | str],
) -> pd.DataFrame:
    """Collect per-object area and bounding-box statistics from labeled masks."""
    rows: list[dict[str, object]] = []

    for segmentation_dir in segmentation_dirs:
        directory = Path(segmentation_dir)
        tif_paths = sorted(directory.glob("*.tif"))

        for tif_path in tif_paths:
            image = tifffile.imread(tif_path)
            object_slices = find_objects(image)

            for label_id, object_slice in enumerate(object_slices, start=1):
                if object_slice is None:
                    continue

                label_mask = image[object_slice] == label_id
                area_px = int(np.count_nonzero(label_mask))
                if area_px == 0:
                    continue

                bbox_height_px = int(object_slice[0].stop - object_slice[0].start)
                bbox_width_px = int(object_slice[1].stop - object_slice[1].start)
                bbox_min_dim_px = min(bbox_height_px, bbox_width_px)
                bbox_max_dim_px = max(bbox_height_px, bbox_width_px)
                bbox_y_start = int(object_slice[0].start)
                bbox_y_end = int(object_slice[0].stop)
                bbox_x_start = int(object_slice[1].start)
                bbox_x_end = int(object_slice[1].stop)

                rows.append(
                    {
                        "directory": str(directory),
                        "directory_name": directory.name,
                        "file_path": str(tif_path),
                        "file_name": tif_path.name,
                        "gt_image": str(tif_path),
                        "source_image": None,
                        "composite_key": None,
                        "campaign_number": None,
                        "time_frame": None,
                        "label_id": label_id,
                        "area_px": area_px,
                        "bbox_y_start": bbox_y_start,
                        "bbox_y_end": bbox_y_end,
                        "bbox_x_start": bbox_x_start,
                        "bbox_x_end": bbox_x_end,
                        "bbox_height_px": bbox_height_px,
                        "bbox_width_px": bbox_width_px,
                        "bbox_max_dim_px": bbox_max_dim_px,
                        "bbox_min_dim_px": bbox_min_dim_px,
                        "bbox_aspect_ratio_wh": float(
                            bbox_width_px / bbox_height_px
                        ),
                        "bbox_elongation_ratio": float(
                            bbox_max_dim_px / bbox_min_dim_px
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _find_project_root(start_path: Path) -> Path:
    """Find the project root by locating the nearest ancestor that contains `data/`."""
    start = start_path if start_path.is_dir() else start_path.parent
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start


def _resolve_dataframe_path(path_value: object, project_root: Path) -> Path | None:
    """Resolve a dataframe path that may be relative to the project root."""
    if path_value is None or pd.isna(path_value):
        return None

    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return project_root / path


def collect_segmentation_object_stats_from_dataframes(
    dataset_dataframe_paths: Sequence[Path | str],
) -> pd.DataFrame:
    """Collect per-object stats from dataset parquets with GT/source image metadata."""
    rows: list[dict[str, object]] = []
    processed_gt_paths: set[str] = set()

    for dataset_dataframe_path in dataset_dataframe_paths:
        parquet_path = Path(dataset_dataframe_path)
        project_root = _find_project_root(parquet_path)
        df = load_dataframe_from_parquet_with_metadata(str(parquet_path))

        required_columns = {"gt_image"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise click.ClickException(
                f"Dataset parquet {parquet_path} is missing required columns: {sorted(missing_columns)}"
            )

        metadata_columns = [
            column
            for column in ["composite_key", "campaign_number", "time_frame", "source_image"]
            if column in df.columns
        ]

        dedup_columns = ["gt_image", *metadata_columns]
        unique_rows = df[dedup_columns].dropna(subset=["gt_image"]).drop_duplicates(
            subset=["gt_image"]
        )

        for row in unique_rows.itertuples(index=False):
            row_data = row._asdict()
            gt_path = _resolve_dataframe_path(row_data["gt_image"], project_root)
            if gt_path is None:
                continue
            gt_path_str = str(gt_path.resolve())
            if gt_path_str in processed_gt_paths:
                continue
            processed_gt_paths.add(gt_path_str)

            if not gt_path.exists():
                continue

            source_path = _resolve_dataframe_path(row_data.get("source_image"), project_root)
            image = tifffile.imread(gt_path)
            object_slices = find_objects(image)

            for label_id, object_slice in enumerate(object_slices, start=1):
                if object_slice is None:
                    continue

                label_mask = image[object_slice] == label_id
                area_px = int(np.count_nonzero(label_mask))
                if area_px == 0:
                    continue

                bbox_y_start = int(object_slice[0].start)
                bbox_y_end = int(object_slice[0].stop)
                bbox_x_start = int(object_slice[1].start)
                bbox_x_end = int(object_slice[1].stop)
                bbox_height_px = bbox_y_end - bbox_y_start
                bbox_width_px = bbox_x_end - bbox_x_start
                bbox_min_dim_px = min(bbox_height_px, bbox_width_px)
                bbox_max_dim_px = max(bbox_height_px, bbox_width_px)

                rows.append(
                    {
                        "dataset_dataframe_path": str(parquet_path),
                        "directory": str(gt_path.parent),
                        "directory_name": gt_path.parent.name,
                        "file_path": str(gt_path),
                        "file_name": gt_path.name,
                        "gt_image": str(gt_path),
                        "source_image": str(source_path) if source_path is not None else None,
                        "composite_key": row_data.get("composite_key"),
                        "campaign_number": row_data.get("campaign_number"),
                        "time_frame": row_data.get("time_frame"),
                        "label_id": label_id,
                        "area_px": area_px,
                        "bbox_y_start": bbox_y_start,
                        "bbox_y_end": bbox_y_end,
                        "bbox_x_start": bbox_x_start,
                        "bbox_x_end": bbox_x_end,
                        "bbox_height_px": bbox_height_px,
                        "bbox_width_px": bbox_width_px,
                        "bbox_max_dim_px": bbox_max_dim_px,
                        "bbox_min_dim_px": bbox_min_dim_px,
                        "bbox_aspect_ratio_wh": float(bbox_width_px / bbox_height_px),
                        "bbox_elongation_ratio": float(
                            bbox_max_dim_px / bbox_min_dim_px
                        ),
                    }
                )

    return pd.DataFrame(rows)


def summarize_segmentation_object_stats(
    stats_df: pd.DataFrame,
    crop_sizes: Iterable[int] = (64,),
    rect_sizes: Iterable[tuple[int, int]] = (),
) -> dict[str, object]:
    """Summarize per-object segmentation statistics and crop-size fit rates."""
    normalized_crop_sizes = tuple(sorted({int(size) for size in crop_sizes}))
    normalized_rect_sizes = tuple(
        sorted({(int(height), int(width)) for height, width in rect_sizes})
    )
    if not normalized_crop_sizes and not normalized_rect_sizes:
        raise ValueError("At least one square crop size or rectangular crop size must be provided.")

    if stats_df.empty:
        return {
            "crop_sizes": list(normalized_crop_sizes),
            "rect_sizes": [
                {"height": height, "width": width} for height, width in normalized_rect_sizes
            ],
            "overall": {"n_files": 0, "n_cells": 0},
            "per_directory": {},
        }

    def _summarize(group_df: pd.DataFrame) -> dict[str, object]:
        summary: dict[str, object] = {
            "n_files": int(group_df["file_path"].nunique()),
            "n_cells": int(len(group_df)),
            "mean_area_px": float(group_df["area_px"].mean()),
            "median_area_px": float(group_df["area_px"].median()),
            "p95_area_px": float(group_df["area_px"].quantile(0.95)),
            "max_area_px": int(group_df["area_px"].max()),
            "mean_bbox_height_px": float(group_df["bbox_height_px"].mean()),
            "p95_bbox_height_px": float(group_df["bbox_height_px"].quantile(0.95)),
            "max_bbox_height_px": int(group_df["bbox_height_px"].max()),
            "mean_bbox_width_px": float(group_df["bbox_width_px"].mean()),
            "p95_bbox_width_px": float(group_df["bbox_width_px"].quantile(0.95)),
            "max_bbox_width_px": int(group_df["bbox_width_px"].max()),
            "max_bbox_dim_px": int(group_df["bbox_max_dim_px"].max()),
            "mean_bbox_aspect_ratio_wh": float(group_df["bbox_aspect_ratio_wh"].mean()),
            "median_bbox_aspect_ratio_wh": float(
                group_df["bbox_aspect_ratio_wh"].median()
            ),
            "p05_bbox_aspect_ratio_wh": float(
                group_df["bbox_aspect_ratio_wh"].quantile(0.05)
            ),
            "p95_bbox_aspect_ratio_wh": float(
                group_df["bbox_aspect_ratio_wh"].quantile(0.95)
            ),
            "mean_bbox_elongation_ratio": float(
                group_df["bbox_elongation_ratio"].mean()
            ),
            "median_bbox_elongation_ratio": float(
                group_df["bbox_elongation_ratio"].median()
            ),
            "p95_bbox_elongation_ratio": float(
                group_df["bbox_elongation_ratio"].quantile(0.95)
            ),
            "max_bbox_elongation_ratio": float(
                group_df["bbox_elongation_ratio"].max()
            ),
            "wide_bbox_rate": float(
                (group_df["bbox_width_px"] > group_df["bbox_height_px"]).mean()
            ),
            "tall_bbox_rate": float(
                (group_df["bbox_height_px"] > group_df["bbox_width_px"]).mean()
            ),
            "square_bbox_rate": float(
                (group_df["bbox_height_px"] == group_df["bbox_width_px"]).mean()
            ),
        }

        for crop_size in normalized_crop_sizes:
            fits_crop = (group_df["bbox_height_px"] <= crop_size) & (
                group_df["bbox_width_px"] <= crop_size
            )
            summary[f"fit_bbox_rate_sz{crop_size}"] = float(fits_crop.mean())
            summary[f"over_bbox_count_sz{crop_size}"] = int((~fits_crop).sum())

        for rect_height, rect_width in normalized_rect_sizes:
            rect_key = f"{rect_height}x{rect_width}"
            fits_rect = (group_df["bbox_height_px"] <= rect_height) & (
                group_df["bbox_width_px"] <= rect_width
            )
            fits_rect_swappable = fits_rect | (
                (group_df["bbox_height_px"] <= rect_width)
                & (group_df["bbox_width_px"] <= rect_height)
            )
            summary[f"fit_bbox_rate_rect{rect_key}"] = float(fits_rect.mean())
            summary[f"over_bbox_count_rect{rect_key}"] = int((~fits_rect).sum())
            summary[f"fit_bbox_rate_rect{rect_key}_swappable"] = float(
                fits_rect_swappable.mean()
            )
            summary[f"over_bbox_count_rect{rect_key}_swappable"] = int(
                (~fits_rect_swappable).sum()
            )

        return summary

    per_directory = {
        directory: _summarize(group_df)
        for directory, group_df in stats_df.groupby("directory", sort=True)
    }

    return {
        "crop_sizes": list(normalized_crop_sizes),
        "rect_sizes": [
            {"height": height, "width": width} for height, width in normalized_rect_sizes
        ],
        "overall": _summarize(stats_df),
        "per_directory": per_directory,
    }


def build_oversized_cell_audit(stats_df: pd.DataFrame, crop_size: int) -> pd.DataFrame:
    """Filter and rank cells whose GT bbox does not fit into the given square crop size."""
    if stats_df.empty:
        return pd.DataFrame()

    oversized = stats_df[
        (stats_df["bbox_height_px"] > crop_size) | (stats_df["bbox_width_px"] > crop_size)
    ].copy()
    if oversized.empty:
        return oversized

    oversized["crop_size"] = int(crop_size)
    oversized["overflow_height_px"] = (
        oversized["bbox_height_px"] - crop_size
    ).clip(lower=0)
    oversized["overflow_width_px"] = (
        oversized["bbox_width_px"] - crop_size
    ).clip(lower=0)
    oversized["bbox_center_y"] = (
        oversized["bbox_y_start"] + oversized["bbox_y_end"]
    ) // 2
    oversized["bbox_center_x"] = (
        oversized["bbox_x_start"] + oversized["bbox_x_end"]
    ) // 2
    oversized["oversize_reason"] = oversized.apply(
        lambda row: (
            "both"
            if row["bbox_height_px"] > crop_size and row["bbox_width_px"] > crop_size
            else "height"
            if row["bbox_height_px"] > crop_size
            else "width"
        ),
        axis=1,
    )
    oversized = oversized.sort_values(
        by=["bbox_max_dim_px", "area_px"], ascending=[False, False]
    ).reset_index(drop=True)
    oversized["rank"] = oversized.index + 1
    return oversized


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale image intensities into uint8 for visualization."""
    array = np.asarray(image)
    if array.ndim > 2:
        array = array[0]
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - min_value) / (max_value - min_value)
    return (scaled * 255.0).clip(0, 255).astype(np.uint8)


def _crop_with_padding(image: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int) -> np.ndarray:
    """Crop an array and pad with zeros when the requested window crosses image bounds."""
    pad_top = max(0, -y_start)
    pad_bottom = max(0, y_end - image.shape[0])
    pad_left = max(0, -x_start)
    pad_right = max(0, x_end - image.shape[1])

    img_y_start = max(0, y_start)
    img_y_end = min(image.shape[0], y_end)
    img_x_start = max(0, x_start)
    img_x_end = min(image.shape[1], x_end)

    cropped = image[img_y_start:img_y_end, img_x_start:img_x_end]
    return np.pad(
        cropped,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
    )


def _draw_rectangle(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    """Draw a rectangle outline with a small fixed width."""
    for inset in range(2):
        draw.rectangle(
            (
                rect[0] - inset,
                rect[1] - inset,
                rect[2] + inset,
                rect[3] + inset,
            ),
            outline=color,
        )


def save_oversized_cell_visualizations(
    audit_df: pd.DataFrame,
    output_dir: Path | str,
    crop_size: int,
    max_visualizations: int = 25,
    context_pad: int = 32,
) -> list[str]:
    """Save raw/GT context PNGs for oversized cells."""
    if audit_df.empty or max_visualizations <= 0:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []

    for row in audit_df.head(max_visualizations).itertuples(index=False):
        gt_path = Path(row.gt_image)
        if not gt_path.exists():
            continue

        gt_image = tifffile.imread(gt_path)
        source_path = Path(row.source_image) if row.source_image else None
        if source_path is not None and source_path.exists():
            raw_image = tifffile.imread(source_path)
        else:
            raw_image = np.zeros_like(gt_image)

        if raw_image.ndim > 2:
            raw_image = raw_image[0]

        center_y = int(row.bbox_center_y)
        center_x = int(row.bbox_center_x)
        context_half_size = max(
            int(crop_size // 2 + context_pad),
            int(row.bbox_max_dim_px // 2 + context_pad),
        )
        context_y_start = center_y - context_half_size
        context_y_end = center_y + context_half_size
        context_x_start = center_x - context_half_size
        context_x_end = center_x + context_half_size

        raw_context = _crop_with_padding(
            raw_image, context_y_start, context_y_end, context_x_start, context_x_end
        )
        gt_context = _crop_with_padding(
            gt_image, context_y_start, context_y_end, context_x_start, context_x_end
        )

        raw_rgb = np.stack([_normalize_to_uint8(raw_context)] * 3, axis=-1)
        overlay_rgb = raw_rgb.copy()
        label_mask = gt_context == int(row.label_id)
        overlay_rgb[label_mask, 0] = 255
        overlay_rgb[label_mask, 1] = (overlay_rgb[label_mask, 1] * 0.35).astype(np.uint8)
        overlay_rgb[label_mask, 2] = (overlay_rgb[label_mask, 2] * 0.35).astype(np.uint8)

        raw_img = Image.fromarray(raw_rgb)
        overlay_img = Image.fromarray(overlay_rgb)
        raw_draw = ImageDraw.Draw(raw_img)
        overlay_draw = ImageDraw.Draw(overlay_img)

        bbox_rect = (
            int(row.bbox_x_start - context_x_start),
            int(row.bbox_y_start - context_y_start),
            int(row.bbox_x_end - context_x_start - 1),
            int(row.bbox_y_end - context_y_start - 1),
        )
        crop_rect = (
            int(center_x - crop_size // 2 - context_x_start),
            int(center_y - crop_size // 2 - context_y_start),
            int(center_x + crop_size // 2 - context_x_start - 1),
            int(center_y + crop_size // 2 - context_y_start - 1),
        )

        _draw_rectangle(raw_draw, bbox_rect, (255, 64, 64))
        _draw_rectangle(raw_draw, crop_rect, (255, 220, 64))
        _draw_rectangle(overlay_draw, bbox_rect, (255, 64, 64))
        _draw_rectangle(overlay_draw, crop_rect, (255, 220, 64))

        spacer = np.full((raw_rgb.shape[0], 12, 3), 255, dtype=np.uint8)
        combined = np.concatenate(
            [np.array(raw_img), spacer, np.array(overlay_img)], axis=1
        )

        composite_key = row.composite_key or gt_path.stem
        output_file = output_path / (
            f"rank{int(row.rank):03d}_{composite_key}_label{int(row.label_id):04d}.png"
        )
        Image.fromarray(combined).save(output_file)
        created_files.append(str(output_file))

    return created_files


def save_segmentation_summary(summary: dict[str, object], output_path: Path | str) -> None:
    """Save a summary dictionary as JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def save_segmentation_object_stats(
    stats_df: pd.DataFrame, output_path: Path | str
) -> None:
    """Save per-object statistics as parquet or CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix == ".parquet":
        stats_df.to_parquet(output, index=False)
        return
    if output.suffix == ".csv":
        stats_df.to_csv(output, index=False)
        return

    raise click.ClickException(
        "Per-object output must end with .parquet or .csv."
    )
