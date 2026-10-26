from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


SQUARE_IMAGE_FEATURE_COLUMNS = [
    "mask_area_pixels",
    "mask_area_fraction",
    "mask_bbox_height",
    "mask_bbox_width",
    "mask_bbox_area",
    "mask_extent",
    "mask_perimeter_pixels",
    "mask_perimeter_area_ratio",
    "mask_component_count",
    "mask_border_touch_fraction",
    "mask_centroid_offset_norm",
    "raw_inside_mean",
    "raw_local_background_mean",
    "raw_contrast",
    "raw_contrast_norm",
    "raw_boundary_mean",
    "raw_boundary_contrast",
]

RECT_IMAGE_FEATURE_COLUMNS = [
    "rect_mask_area_pixels",
    "rect_mask_area_fraction",
    "rect_mask_bbox_height",
    "rect_mask_bbox_width",
    "rect_mask_bbox_aspect_ratio",
    "rect_mask_bbox_area",
    "rect_mask_extent",
    "rect_mask_perimeter_pixels",
    "rect_mask_perimeter_area_ratio",
    "rect_mask_component_count",
    "rect_mask_border_touch_fraction",
    "rect_mask_centroid_offset_norm",
    "rect_raw_inside_mean",
    "rect_raw_local_background_mean",
    "rect_raw_contrast",
    "rect_raw_contrast_norm",
    "rect_raw_boundary_mean",
    "rect_raw_boundary_contrast",
]

MULTISCALE_RECT_IMAGE_FEATURE_COLUMNS = [
    f"{prefix}_{name}"
    for prefix in ("tight", "far")
    for name in [
        "mask_area_pixels",
        "mask_area_fraction",
        "mask_bbox_height",
        "mask_bbox_width",
        "mask_bbox_aspect_ratio",
        "mask_bbox_area",
        "mask_extent",
        "mask_perimeter_pixels",
        "mask_perimeter_area_ratio",
        "mask_component_count",
        "mask_border_touch_fraction",
        "mask_centroid_offset_norm",
        "raw_inside_mean",
        "raw_local_background_mean",
        "raw_contrast",
        "raw_contrast_norm",
        "raw_boundary_mean",
        "raw_boundary_contrast",
    ]
]

FULL_IMAGE_BASE_FEATURE_COLUMNS = [
    "full_mask_area_pixels",
    "full_mask_area_fraction",
    "full_mask_retained_fraction",
    "full_mask_bbox_height",
    "full_mask_bbox_width",
    "full_mask_bbox_aspect_ratio",
    "full_mask_bbox_area",
    "full_mask_extent",
    "full_mask_perimeter_pixels",
    "full_mask_perimeter_area_ratio",
    "full_mask_component_count",
    "full_mask_border_touch_fraction",
    "full_mask_centroid_offset_norm",
    "full_raw_inside_mean",
    "full_raw_local_background_mean",
    "full_raw_contrast",
    "full_raw_contrast_norm",
    "full_raw_boundary_mean",
    "full_raw_boundary_contrast",
]

FULL_IMAGE_BOUNDARY_NOISE_FEATURE_COLUMNS = [
    "full_raw_background_std",
    "full_raw_inside_std",
    "full_raw_contrast_to_noise",
    "full_raw_inside_p25",
    "full_raw_inside_p75",
    "full_raw_background_p25",
    "full_raw_background_p75",
    "full_raw_boundary_gradient_mean",
    "full_raw_boundary_gradient_contrast",
    "full_mask_hole_fraction",
    "full_mask_eccentricity",
]

FULL_IMAGE_FEATURE_COLUMNS = (
    FULL_IMAGE_BASE_FEATURE_COLUMNS + FULL_IMAGE_BOUNDARY_NOISE_FEATURE_COLUMNS
)

CROP_GEOMETRY_FEATURE_COLUMNS = [
    "crop_overflow_top",
    "crop_overflow_bottom",
    "crop_overflow_left",
    "crop_overflow_right",
    "crop_visible_height",
    "crop_visible_width",
    "crop_visible_fraction",
    "crop_has_padding",
    "mask_touching_square_crop_border",
    "mask_touching_rect_border",
    "mask_touching_image_border",
]


FEATURE_COLUMNS = [
    "source_path",
    "dataset",
    "crop_tag",
    "variant",
    "fold",
    "candidate_id",
    "image_id",
    "split",
    "campaign_number",
    "original_image_key",
    "time_frame",
    "candidate_source_raw",
    "candidate_source",
    "label",
    "crop_y_start",
    "crop_y_end",
    "crop_x_start",
    "crop_x_end",
    "crop_width",
    "crop_height",
    "crop_area",
    "center_y",
    "center_x",
    "fusion_support",
    "fusion_support_rank_in_image",
    "qa_score",
    "qa_rank_in_image",
    "true_iou",
    "true_f1",
    "source_train_mean_iou",
    "source_train_n",
    "dataset_train_mean_iou",
    "temporal_prev_min_center_dist",
    "temporal_next_min_center_dist",
    "temporal_min_center_dist",
    "temporal_min_center_dist_norm",
    "temporal_prev_support_score",
    "temporal_next_support_score",
    "temporal_support_score",
    "source_temporal_prev_min_center_dist",
    "source_temporal_next_min_center_dist",
    "source_temporal_min_center_dist",
    "source_temporal_min_center_dist_norm",
    "source_temporal_prev_support_score",
    "source_temporal_next_support_score",
    "source_temporal_support_score",
] + SQUARE_IMAGE_FEATURE_COLUMNS + RECT_IMAGE_FEATURE_COLUMNS + MULTISCALE_RECT_IMAGE_FEATURE_COLUMNS + FULL_IMAGE_FEATURE_COLUMNS + CROP_GEOMETRY_FEATURE_COLUMNS


def build_candidate_feature_table(
    input_paths: Iterable[Path],
    *,
    include_image_features: bool = False,
    include_rect_image_features: bool = False,
    include_multiscale_rect_image_features: bool = False,
    include_full_image_features: bool = False,
    include_full_image_boundary_noise_features: bool = False,
    include_crop_geometry_features: bool = False,
) -> pd.DataFrame:
    frames = [
        build_candidate_features(
            pd.read_parquet(path),
            source_path=path,
            include_image_features=include_image_features,
            include_rect_image_features=include_rect_image_features,
            include_multiscale_rect_image_features=(
                include_multiscale_rect_image_features
            ),
            include_full_image_features=include_full_image_features,
            include_full_image_boundary_noise_features=(
                include_full_image_boundary_noise_features
            ),
            include_crop_geometry_features=include_crop_geometry_features,
        )
        for path in input_paths
    ]
    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.concat(frames, ignore_index=True)[FEATURE_COLUMNS]


def build_candidate_features(
    candidates: pd.DataFrame,
    *,
    source_path: Path | None = None,
    include_image_features: bool = False,
    include_rect_image_features: bool = False,
    include_multiscale_rect_image_features: bool = False,
    include_full_image_features: bool = False,
    include_full_image_boundary_noise_features: bool = False,
    include_crop_geometry_features: bool = False,
) -> pd.DataFrame:
    required = {
        "cell_id",
        "original_image_key",
        "campaign_number",
        "competitor",
        "label",
        "crop_y_start",
        "crop_y_end",
        "crop_x_start",
        "crop_x_end",
        "original_center_y",
        "original_center_x",
        "center_agreement_count",
        "split",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidate parquet is missing columns: {missing}")

    metadata = _parse_paper_input_path(source_path)
    features = pd.DataFrame(index=candidates.index)
    features["source_path"] = str(source_path) if source_path is not None else ""
    for key, value in metadata.items():
        features[key] = value

    features["candidate_id"] = candidates["cell_id"].astype(str)
    features["image_id"] = candidates.apply(_image_id, axis=1)
    features["split"] = candidates["split"].astype(str)
    features["campaign_number"] = candidates["campaign_number"].map(_campaign_label)
    features["original_image_key"] = candidates["original_image_key"].astype(str)
    features["time_frame"] = _numeric(candidates.get("time_frame"))
    features["candidate_source_raw"] = candidates["competitor"].astype(str)
    features["candidate_source"] = features["candidate_source_raw"].map(
        canonicalize_candidate_source
    )
    features["label"] = candidates["label"].astype(str)

    y_start = _numeric(candidates["crop_y_start"])
    y_end = _numeric(candidates["crop_y_end"])
    x_start = _numeric(candidates["crop_x_start"])
    x_end = _numeric(candidates["crop_x_end"])
    features["crop_y_start"] = y_start
    features["crop_y_end"] = y_end
    features["crop_x_start"] = x_start
    features["crop_x_end"] = x_end
    features["crop_width"] = (x_end - x_start).clip(lower=0)
    features["crop_height"] = (y_end - y_start).clip(lower=0)
    features["crop_area"] = features["crop_width"] * features["crop_height"]
    features["center_y"] = _numeric(candidates["original_center_y"])
    features["center_x"] = _numeric(candidates["original_center_x"])

    features["fusion_support"] = _numeric(candidates["center_agreement_count"])
    features["fusion_support_rank_in_image"] = _rank_within_image(
        features, "fusion_support"
    )
    features["qa_score"] = _optional_numeric(candidates, "predicted_jaccard_index")
    features["qa_rank_in_image"] = _rank_within_image(features, "qa_score")
    features["true_iou"] = _optional_numeric(candidates, "jaccard_score")
    features["true_f1"] = _optional_numeric(candidates, "f1_score")

    _add_source_train_priors(features)
    _add_temporal_distance_features(features, source_column=None, prefix="temporal")
    _add_temporal_distance_features(
        features, source_column="candidate_source", prefix="source_temporal"
    )
    features = _add_image_mask_features(
        features,
        candidates,
        include_square=include_image_features,
        include_rect=include_rect_image_features,
        include_multiscale_rect=include_multiscale_rect_image_features,
    )
    features = _add_crop_geometry_features(
        features,
        candidates,
        enabled=include_crop_geometry_features,
    )
    features = _add_full_image_features(
        features,
        candidates,
        enabled=include_full_image_features or include_full_image_boundary_noise_features,
        include_boundary_noise=include_full_image_boundary_noise_features,
    )

    return features[FEATURE_COLUMNS]


def canonicalize_candidate_source(source: object) -> str:
    text = str(source)
    alias_map = {
        "BGU-IL (5)": "BGU-IL-5",
    }
    return alias_map.get(text, text)


def write_candidate_feature_table(
    input_paths: Iterable[Path],
    output_path: Path,
    *,
    include_image_features: bool = False,
    include_rect_image_features: bool = False,
    include_multiscale_rect_image_features: bool = False,
    include_full_image_features: bool = False,
    include_full_image_boundary_noise_features: bool = False,
    include_crop_geometry_features: bool = False,
) -> pd.DataFrame:
    features = build_candidate_feature_table(
        input_paths,
        include_image_features=include_image_features,
        include_rect_image_features=include_rect_image_features,
        include_multiscale_rect_image_features=include_multiscale_rect_image_features,
        include_full_image_features=include_full_image_features,
        include_full_image_boundary_noise_features=(
            include_full_image_boundary_noise_features
        ),
        include_crop_geometry_features=include_crop_geometry_features,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        features.to_parquet(output_path, index=False)
    else:
        features.to_csv(output_path, index=False)
    return features


def _parse_paper_input_path(path: Path | None) -> dict[str, str]:
    metadata = {"dataset": "", "crop_tag": "", "variant": "", "fold": ""}
    if path is None:
        return metadata

    parts = path.parts
    if "paper_inputs" not in parts:
        return metadata
    input_index = parts.index("paper_inputs")
    if len(parts) <= input_index + 4:
        return metadata

    metadata["dataset"] = parts[input_index + 1]
    metadata["crop_tag"] = parts[input_index + 2]
    metadata["variant"] = parts[input_index + 3]
    match = re.match(r"(fold-\d+)_", path.stem)
    if match:
        metadata["fold"] = match.group(1)
    return metadata


def _campaign_label(value: object) -> str:
    text = str(value)
    match = re.search(r"\d+", text)
    if match is None:
        return text
    return match.group(0).zfill(2)


def _image_id(row: pd.Series) -> str:
    return f"{_campaign_label(row['campaign_number'])}_{row['original_image_key']}"


def _numeric(value: object) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    return pd.Series(np.nan)


def _optional_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _rank_within_image(features: pd.DataFrame, column: str) -> pd.Series:
    return features.groupby("image_id", dropna=False)[column].rank(
        method="average", pct=True
    )


def _add_source_train_priors(features: pd.DataFrame) -> None:
    train_rows = features[
        features["split"].eq("train") & features["true_iou"].notna()
    ].copy()
    if train_rows.empty:
        features["source_train_mean_iou"] = np.nan
        features["source_train_n"] = 0
        features["dataset_train_mean_iou"] = np.nan
        return

    source_priors = train_rows.groupby("candidate_source")["true_iou"].agg(
        ["mean", "count"]
    )
    features["source_train_mean_iou"] = features["candidate_source"].map(
        source_priors["mean"]
    )
    features["source_train_n"] = (
        features["candidate_source"].map(source_priors["count"]).fillna(0).astype(int)
    )
    features["dataset_train_mean_iou"] = float(train_rows["true_iou"].mean())


def _add_temporal_distance_features(
    features: pd.DataFrame,
    *,
    source_column: str | None,
    prefix: str,
) -> None:
    prev_values = pd.Series(np.nan, index=features.index, dtype=float)
    next_values = pd.Series(np.nan, index=features.index, dtype=float)
    prev_support = pd.Series(np.nan, index=features.index, dtype=float)
    next_support = pd.Series(np.nan, index=features.index, dtype=float)
    neighbor_quality = _temporal_neighbor_quality(features)

    group_columns = ["campaign_number"]
    if source_column is not None:
        group_columns.append(source_column)

    for _, group in features.groupby(group_columns, dropna=False):
        frame_points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for frame, frame_group in group.dropna(subset=["time_frame"]).groupby(
            "time_frame"
        ):
            frame_points[int(frame)] = (
                frame_group[["center_y", "center_x"]].to_numpy(dtype=float),
                neighbor_quality.loc[frame_group.index].to_numpy(dtype=float),
            )

        for index, row in group.iterrows():
            if pd.isna(row["time_frame"]) or pd.isna(row["center_y"]) or pd.isna(
                row["center_x"]
            ):
                continue
            point = np.array([float(row["center_y"]), float(row["center_x"])])
            frame = int(row["time_frame"])
            crop_size = float(np.sqrt(row["crop_area"])) if row["crop_area"] > 0 else np.nan
            prev_data = frame_points.get(frame - 1)
            next_data = frame_points.get(frame + 1)
            prev_values.at[index] = _nearest_distance(point, prev_data)
            next_values.at[index] = _nearest_distance(point, next_data)
            prev_support.at[index] = _temporal_support_score(
                point, crop_size, prev_data
            )
            next_support.at[index] = _temporal_support_score(
                point, crop_size, next_data
            )

    min_values = pd.concat([prev_values, next_values], axis=1).min(axis=1)
    support_values = pd.concat([prev_support, next_support], axis=1).max(axis=1)
    crop_size = np.sqrt(features["crop_area"].replace(0, np.nan))
    features[f"{prefix}_prev_min_center_dist"] = prev_values
    features[f"{prefix}_next_min_center_dist"] = next_values
    features[f"{prefix}_min_center_dist"] = min_values
    features[f"{prefix}_min_center_dist_norm"] = min_values / crop_size
    features[f"{prefix}_prev_support_score"] = prev_support
    features[f"{prefix}_next_support_score"] = next_support
    features[f"{prefix}_support_score"] = support_values


def _temporal_neighbor_quality(features: pd.DataFrame) -> pd.Series:
    quality = features[["fusion_support_rank_in_image", "qa_rank_in_image"]].mean(
        axis=1, skipna=True
    )
    return quality.fillna(0.5).clip(lower=0.0, upper=1.0)


def _nearest_distance(
    point: np.ndarray, candidates: tuple[np.ndarray, np.ndarray] | None
) -> float:
    if candidates is None or len(candidates[0]) == 0:
        return float("nan")
    distances = np.linalg.norm(candidates[0] - point, axis=1)
    return float(np.min(distances))


def _temporal_support_score(
    point: np.ndarray,
    crop_size: float,
    candidates: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    if candidates is None or len(candidates[0]) == 0:
        return float("nan")
    points, qualities = candidates
    distances = np.linalg.norm(points - point, axis=1)
    if np.isfinite(crop_size) and crop_size > 0:
        distances = distances / crop_size
    proximity = np.exp(-distances)
    scores = proximity * np.nan_to_num(qualities, nan=0.5)
    return float(np.clip(np.max(scores), 0.0, 1.0))


def _add_image_mask_features(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    include_square: bool,
    include_rect: bool,
    include_multiscale_rect: bool,
) -> pd.DataFrame:
    features = _with_nan_columns(
        features,
        SQUARE_IMAGE_FEATURE_COLUMNS
        + RECT_IMAGE_FEATURE_COLUMNS
        + MULTISCALE_RECT_IMAGE_FEATURE_COLUMNS,
    )

    if (
        not include_square
        and not include_rect
        and not include_multiscale_rect
        or "stacked_path" not in candidates.columns
    ):
        return features

    for index, path_value in candidates["stacked_path"].items():
        if pd.isna(path_value):
            continue
        record = _image_mask_feature_record(
            Path(str(path_value)),
            include_square=include_square,
            include_rect=include_rect,
            include_multiscale_rect=include_multiscale_rect,
        )
        for column, value in record.items():
            features.at[index, column] = value
    return features


def _image_mask_feature_record(
    path: Path,
    *,
    include_square: bool,
    include_rect: bool,
    include_multiscale_rect: bool,
) -> dict[str, float]:
    if not path.exists():
        return {}
    image = tifffile.imread(path)
    if image.ndim == 3:
        raw = image[0].astype(float, copy=False)
        mask = image[1] > 0
    elif image.ndim == 2:
        raw = image.astype(float, copy=False)
        mask = image > 0
    else:
        return {}

    if mask.size == 0:
        return {}

    record: dict[str, float] = {}
    if include_square:
        record.update(_mask_raw_feature_record(raw, mask, prefix=""))
    if include_rect:
        rectangle = _expanded_mask_rectangle(mask)
        if rectangle is not None:
            y0, y1, x0, x1 = rectangle
            record.update(
                _mask_raw_feature_record(
                    raw[y0:y1, x0:x1],
                    mask[y0:y1, x0:x1],
                    prefix="rect_",
                )
            )
    if include_multiscale_rect:
        for prefix, min_margin, margin_fraction in (
            ("tight_", 0, 0.0),
            ("far_", 32, 0.75),
        ):
            rectangle = _expanded_mask_rectangle(
                mask, min_margin=min_margin, margin_fraction=margin_fraction
            )
            if rectangle is None:
                continue
            y0, y1, x0, x1 = rectangle
            record.update(
                _mask_raw_feature_record(
                    raw[y0:y1, x0:x1],
                    mask[y0:y1, x0:x1],
                    prefix=prefix,
                )
            )
    return record


def _expanded_mask_rectangle(
    mask: np.ndarray,
    *,
    min_margin: int = 16,
    margin_fraction: float = 0.25,
) -> tuple[int, int, int, int] | None:
    area = float(mask.sum())
    if area <= 0:
        return None

    rows, cols = np.nonzero(mask)
    bbox_height = int(rows.max() - rows.min() + 1)
    bbox_width = int(cols.max() - cols.min() + 1)
    margin = int(np.ceil(max(float(min_margin), margin_fraction * max(bbox_height, bbox_width))))
    y0 = max(0, int(rows.min()) - margin)
    y1 = min(mask.shape[0], int(rows.max()) + margin + 1)
    x0 = max(0, int(cols.min()) - margin)
    x1 = min(mask.shape[1], int(cols.max()) + margin + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def _mask_raw_feature_record(
    raw: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str,
    include_full_boundary_noise: bool = False,
) -> dict[str, float]:
    area = float(mask.sum())
    area_fraction = float(area / mask.size) if mask.size > 0 else np.nan
    if area <= 0:
        return {
            f"{prefix}mask_area_pixels": 0.0,
            f"{prefix}mask_area_fraction": 0.0,
            f"{prefix}mask_component_count": 0.0,
        }

    rows, cols = np.nonzero(mask)
    bbox_height = float(rows.max() - rows.min() + 1)
    bbox_width = float(cols.max() - cols.min() + 1)
    bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else np.nan
    bbox_area = bbox_height * bbox_width
    extent = float(area / bbox_area) if bbox_area > 0 else np.nan

    eroded = ndimage.binary_erosion(mask)
    perimeter = float(np.logical_and(mask, ~eroded).sum())
    labeled, component_count = ndimage.label(mask)
    del labeled

    border_pixels = (
        float(mask[0, :].sum())
        + float(mask[-1, :].sum())
        + float(mask[:, 0].sum())
        + float(mask[:, -1].sum())
    )
    border_touch_fraction = border_pixels / max(perimeter, 1.0)

    centroid_y = float(rows.mean())
    centroid_x = float(cols.mean())
    center_y = (mask.shape[0] - 1) / 2.0
    center_x = (mask.shape[1] - 1) / 2.0
    diagonal = float(np.hypot(mask.shape[0], mask.shape[1]))
    centroid_offset = float(np.hypot(centroid_y - center_y, centroid_x - center_x))
    centroid_offset_norm = centroid_offset / diagonal if diagonal > 0 else np.nan

    dilated = ndimage.binary_dilation(mask, iterations=3)
    boundary_ring = np.logical_and(dilated, ~mask)
    inside = raw[mask]
    background = raw[boundary_ring]
    raw_inside_mean = _safe_mean(inside)
    raw_background_mean = _safe_mean(background)
    raw_background_std = _safe_std(background)
    raw_contrast = raw_inside_mean - raw_background_mean
    raw_contrast_norm = raw_contrast / raw_background_std if raw_background_std > 0 else np.nan

    boundary = np.logical_and(mask, ~eroded)
    raw_boundary_mean = _safe_mean(raw[boundary])
    raw_boundary_contrast = raw_boundary_mean - raw_background_mean

    record = {
        f"{prefix}mask_area_pixels": area,
        f"{prefix}mask_area_fraction": area_fraction,
        f"{prefix}mask_bbox_height": bbox_height,
        f"{prefix}mask_bbox_width": bbox_width,
        f"{prefix}mask_bbox_area": bbox_area,
        f"{prefix}mask_extent": extent,
        f"{prefix}mask_perimeter_pixels": perimeter,
        f"{prefix}mask_perimeter_area_ratio": perimeter / area,
        f"{prefix}mask_component_count": float(component_count),
        f"{prefix}mask_border_touch_fraction": border_touch_fraction,
        f"{prefix}mask_centroid_offset_norm": centroid_offset_norm,
        f"{prefix}raw_inside_mean": raw_inside_mean,
        f"{prefix}raw_local_background_mean": raw_background_mean,
        f"{prefix}raw_contrast": raw_contrast,
        f"{prefix}raw_contrast_norm": raw_contrast_norm,
        f"{prefix}raw_boundary_mean": raw_boundary_mean,
        f"{prefix}raw_boundary_contrast": raw_boundary_contrast,
    }
    if prefix:
        record[f"{prefix}mask_bbox_aspect_ratio"] = bbox_aspect_ratio
    if prefix == "full_" and include_full_boundary_noise:
        record.update(
            _full_image_boundary_noise_feature_record(
                raw=raw,
                mask=mask,
                boundary=boundary,
                boundary_ring=boundary_ring,
                inside=inside,
                background=background,
                inside_mean=raw_inside_mean,
                background_mean=raw_background_mean,
                background_std=raw_background_std,
                rows=rows,
                cols=cols,
                area=area,
            )
        )
    return record


def _full_image_boundary_noise_feature_record(
    *,
    raw: np.ndarray,
    mask: np.ndarray,
    boundary: np.ndarray,
    boundary_ring: np.ndarray,
    inside: np.ndarray,
    background: np.ndarray,
    inside_mean: float,
    background_mean: float,
    background_std: float,
    rows: np.ndarray,
    cols: np.ndarray,
    area: float,
) -> dict[str, float]:
    return {
        "full_raw_background_std": background_std,
        "full_raw_inside_std": _safe_std(inside),
        "full_raw_contrast_to_noise": (
            (inside_mean - background_mean) / (background_std + 1e-6)
            if np.isfinite(inside_mean) and np.isfinite(background_mean)
            else np.nan
        ),
        "full_raw_inside_p25": _safe_percentile(inside, 25),
        "full_raw_inside_p75": _safe_percentile(inside, 75),
        "full_raw_background_p25": _safe_percentile(background, 25),
        "full_raw_background_p75": _safe_percentile(background, 75),
        "full_raw_boundary_gradient_mean": _boundary_gradient_mean(
            raw, boundary
        ),
        "full_raw_boundary_gradient_contrast": _boundary_gradient_contrast(
            raw, boundary, boundary_ring
        ),
        "full_mask_hole_fraction": _mask_hole_fraction(mask, area=area),
        "full_mask_eccentricity": _mask_eccentricity(rows, cols),
    }


def _add_crop_geometry_features(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    enabled: bool,
) -> pd.DataFrame:
    features = _with_nan_columns(features, CROP_GEOMETRY_FEATURE_COLUMNS)

    if not enabled:
        return features

    shape_cache: dict[str, tuple[int, int] | None] = {}
    for index, row in candidates.iterrows():
        image_shape = _cached_image_shape(row.get("original_image_path"), shape_cache)
        record = _crop_geometry_feature_record(row, image_shape=image_shape)
        if "stacked_path" in candidates.columns:
            record.update(
                _mask_border_geometry_record(
                    row,
                    image_shape=image_shape,
                    stacked_path_value=row.get("stacked_path"),
                )
        )
        for column, value in record.items():
            features.at[index, column] = value
    return features


def _with_nan_columns(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in features.columns]
    if not missing:
        return features
    nan_columns = pd.DataFrame(np.nan, index=features.index, columns=missing)
    return pd.concat([features, nan_columns], axis=1)


def _add_full_image_features(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    enabled: bool,
    include_boundary_noise: bool,
) -> pd.DataFrame:
    features = _with_nan_columns(features, FULL_IMAGE_FEATURE_COLUMNS)
    if (
        not enabled
        or "stacked_path" not in candidates.columns
        or "original_image_path" not in candidates.columns
    ):
        return features

    raw_cache: dict[str, np.ndarray | None] = {}
    for index, row in candidates.iterrows():
        record = _full_image_feature_record(
            row,
            raw_cache=raw_cache,
            include_boundary_noise=include_boundary_noise,
        )
        for column, value in record.items():
            features.at[index, column] = value
    return features


def _full_image_feature_record(
    row: pd.Series,
    *,
    raw_cache: dict[str, np.ndarray | None],
    include_boundary_noise: bool,
) -> dict[str, float]:
    stacked_path_value = row.get("stacked_path")
    original_path_value = row.get("original_image_path")
    if pd.isna(stacked_path_value) or pd.isna(original_path_value):
        return {}

    mask = _read_stacked_mask(Path(str(stacked_path_value)))
    raw = _cached_raw_image(original_path_value, raw_cache)
    if mask is None or raw is None or mask.size == 0:
        return {}

    rectangle = _expanded_mask_rectangle(mask)
    if rectangle is None:
        return {
            "full_mask_area_pixels": 0.0,
            "full_mask_area_fraction": 0.0,
            "full_mask_retained_fraction": 0.0,
            "full_mask_component_count": 0.0,
        }
    if pd.isna(row.get("crop_y_start")) or pd.isna(row.get("crop_x_start")):
        return {}

    crop_y_start = int(float(row.get("crop_y_start")))
    crop_x_start = int(float(row.get("crop_x_start")))
    y0, y1, x0, x1 = rectangle
    original_y0 = max(0, crop_y_start + y0)
    original_y1 = min(raw.shape[0], crop_y_start + y1)
    original_x0 = max(0, crop_x_start + x0)
    original_x1 = min(raw.shape[1], crop_x_start + x1)
    if original_y1 <= original_y0 or original_x1 <= original_x0:
        return {
            "full_mask_area_pixels": 0.0,
            "full_mask_area_fraction": 0.0,
            "full_mask_retained_fraction": 0.0,
            "full_mask_component_count": 0.0,
        }

    mask_y0 = original_y0 - crop_y_start
    mask_y1 = original_y1 - crop_y_start
    mask_x0 = original_x0 - crop_x_start
    mask_x1 = original_x1 - crop_x_start
    raw_patch = raw[original_y0:original_y1, original_x0:original_x1]
    mask_patch = mask[mask_y0:mask_y1, mask_x0:mask_x1]
    if raw_patch.shape != mask_patch.shape:
        return {}

    record = _mask_raw_feature_record(
        raw_patch,
        mask_patch,
        prefix="full_",
        include_full_boundary_noise=include_boundary_noise,
    )
    original_area = float(mask.sum())
    retained_area = float(mask_patch.sum())
    record["full_mask_retained_fraction"] = (
        retained_area / original_area if original_area > 0 else np.nan
    )
    return record


def _read_stacked_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        image = tifffile.imread(path)
    except (OSError, ValueError):
        return None
    if image.ndim == 3:
        return image[1] > 0
    if image.ndim == 2:
        return image > 0
    return None


def _cached_raw_image(
    path_value: object, cache: dict[str, np.ndarray | None]
) -> np.ndarray | None:
    if pd.isna(path_value):
        return None
    path_text = str(path_value)
    if path_text not in cache:
        cache[path_text] = _read_raw_image(Path(path_text))
    return cache[path_text]


def _read_raw_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        image = tifffile.imread(path)
    except (OSError, ValueError):
        return None
    raw = np.asarray(image)
    raw = np.squeeze(raw)
    if raw.ndim > 2:
        raw = raw[0]
    if raw.ndim != 2:
        return None
    return raw.astype(float, copy=False)


def _cached_image_shape(
    path_value: object, cache: dict[str, tuple[int, int] | None]
) -> tuple[int, int] | None:
    if pd.isna(path_value):
        return None
    path_text = str(path_value)
    if path_text not in cache:
        cache[path_text] = _read_image_shape(Path(path_text))
    return cache[path_text]


def _read_image_shape(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    try:
        with tifffile.TiffFile(path) as tif:
            shape = tif.pages[0].shape
    except (OSError, ValueError, IndexError):
        return None
    if len(shape) < 2:
        return None
    return int(shape[-2]), int(shape[-1])


def _crop_geometry_feature_record(
    row: pd.Series, *, image_shape: tuple[int, int] | None
) -> dict[str, float]:
    values = [
        row.get("crop_y_start"),
        row.get("crop_y_end"),
        row.get("crop_x_start"),
        row.get("crop_x_end"),
    ]
    if image_shape is None or any(pd.isna(value) for value in values):
        return {}

    y_start, y_end, x_start, x_end = [float(value) for value in values]
    image_height, image_width = image_shape
    crop_height = max(0.0, y_end - y_start)
    crop_width = max(0.0, x_end - x_start)
    crop_area = crop_height * crop_width

    overflow_top = max(0.0, -y_start)
    overflow_bottom = max(0.0, y_end - image_height)
    overflow_left = max(0.0, -x_start)
    overflow_right = max(0.0, x_end - image_width)
    visible_height = max(0.0, min(y_end, image_height) - max(y_start, 0.0))
    visible_width = max(0.0, min(x_end, image_width) - max(x_start, 0.0))
    visible_fraction = (
        (visible_height * visible_width) / crop_area if crop_area > 0 else np.nan
    )

    return {
        "crop_overflow_top": overflow_top,
        "crop_overflow_bottom": overflow_bottom,
        "crop_overflow_left": overflow_left,
        "crop_overflow_right": overflow_right,
        "crop_visible_height": visible_height,
        "crop_visible_width": visible_width,
        "crop_visible_fraction": visible_fraction,
        "crop_has_padding": float(
            overflow_top > 0
            or overflow_bottom > 0
            or overflow_left > 0
            or overflow_right > 0
        ),
    }


def _mask_border_geometry_record(
    row: pd.Series,
    *,
    image_shape: tuple[int, int] | None,
    stacked_path_value: object,
) -> dict[str, float]:
    if pd.isna(stacked_path_value):
        return {}
    path = Path(str(stacked_path_value))
    if not path.exists():
        return {}
    try:
        image = tifffile.imread(path)
    except (OSError, ValueError):
        return {}
    if image.ndim == 3:
        mask = image[1] > 0
    elif image.ndim == 2:
        mask = image > 0
    else:
        return {}
    if mask.size == 0 or not mask.any():
        return {
            "mask_touching_square_crop_border": 0.0,
            "mask_touching_rect_border": 0.0,
            "mask_touching_image_border": 0.0,
        }

    rows, cols = np.nonzero(mask)
    touches_square = (
        rows.min() == 0
        or rows.max() == mask.shape[0] - 1
        or cols.min() == 0
        or cols.max() == mask.shape[1] - 1
    )
    rectangle = _expanded_mask_rectangle(mask)
    touches_rect = False
    if rectangle is not None:
        y0, y1, x0, x1 = rectangle
        rect_mask = mask[y0:y1, x0:x1]
        touches_rect = bool(
            rect_mask[0, :].any()
            or rect_mask[-1, :].any()
            or rect_mask[:, 0].any()
            or rect_mask[:, -1].any()
        )

    touches_image = False
    if image_shape is not None and not pd.isna(row.get("crop_y_start")) and not pd.isna(
        row.get("crop_x_start")
    ):
        image_height, image_width = image_shape
        original_rows = rows + int(float(row.get("crop_y_start")))
        original_cols = cols + int(float(row.get("crop_x_start")))
        touches_image = bool(
            original_rows.min() <= 0
            or original_rows.max() >= image_height - 1
            or original_cols.min() <= 0
            or original_cols.max() >= image_width - 1
        )

    return {
        "mask_touching_square_crop_border": float(touches_square),
        "mask_touching_rect_border": float(touches_rect),
        "mask_touching_image_border": float(touches_image),
    }


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def _gradient_magnitude(raw: np.ndarray) -> np.ndarray | None:
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2:
        return None
    grad_y, grad_x = np.gradient(raw.astype(float, copy=False))
    return np.hypot(grad_y, grad_x)


def _boundary_gradient_mean(raw: np.ndarray, boundary: np.ndarray) -> float:
    gradient = _gradient_magnitude(raw)
    if gradient is None:
        return float("nan")
    return _safe_mean(gradient[boundary])


def _boundary_gradient_contrast(
    raw: np.ndarray, boundary: np.ndarray, boundary_ring: np.ndarray
) -> float:
    gradient = _gradient_magnitude(raw)
    if gradient is None:
        return float("nan")
    return _safe_mean(gradient[boundary]) - _safe_mean(gradient[boundary_ring])


def _mask_hole_fraction(mask: np.ndarray, *, area: float) -> float:
    if area <= 0:
        return 0.0
    filled = ndimage.binary_fill_holes(mask)
    hole_area = float(filled.sum()) - area
    return hole_area / max(area, 1.0)


def _mask_eccentricity(rows: np.ndarray, cols: np.ndarray) -> float:
    if rows.size < 3:
        return float("nan")
    coords = np.vstack([rows.astype(float), cols.astype(float)])
    covariance = np.cov(coords)
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(np.max(eigenvalues))
    smallest = float(np.min(eigenvalues))
    if largest <= 0:
        return float("nan")
    return float(np.sqrt(max(0.0, 1.0 - (smallest / largest))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export candidate-level features for soft hybrid/ranker experiments."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more paper-ready candidate parquet files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output CSV or Parquet feature table path.",
    )
    parser.add_argument(
        "--include-image-features",
        action="store_true",
        help="Read stacked QA crop TIFFs and add mask/image evidence features.",
    )
    parser.add_argument(
        "--include-rect-image-features",
        action="store_true",
        help=(
            "Read stacked QA crop TIFFs and add object-local rectangular "
            "mask/image evidence features."
        ),
    )
    parser.add_argument(
        "--include-multiscale-rect-image-features",
        action="store_true",
        help=(
            "Read stacked QA crop TIFFs and add tight/far rectangular "
            "mask/image evidence features."
        ),
    )
    parser.add_argument(
        "--include-full-image-features",
        action="store_true",
        help=(
            "Read original images and add object-local mask/image evidence "
            "features in true image coordinates."
        ),
    )
    parser.add_argument(
        "--include-full-image-boundary-noise-features",
        action="store_true",
        help=(
            "Read original images and add full-image boundary/noise diagnostics. "
            "This also enables base full-image feature extraction."
        ),
    )
    parser.add_argument(
        "--include-crop-geometry-features",
        action="store_true",
        help=(
            "Read original image shapes and add crop padding/image-border "
            "geometry features."
        ),
    )
    args = parser.parse_args()

    features = write_candidate_feature_table(
        args.inputs,
        args.output,
        include_image_features=args.include_image_features,
        include_rect_image_features=args.include_rect_image_features,
        include_multiscale_rect_image_features=(
            args.include_multiscale_rect_image_features
        ),
        include_full_image_features=args.include_full_image_features,
        include_full_image_boundary_noise_features=(
            args.include_full_image_boundary_noise_features
        ),
        include_crop_geometry_features=args.include_crop_geometry_features,
    )
    print(f"Wrote {len(features)} candidate feature rows to {args.output}")


if __name__ == "__main__":
    main()
