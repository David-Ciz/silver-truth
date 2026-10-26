from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SoftHybridWeights:
    fusion_support: float = 0.40
    qa: float = 0.25
    temporal: float = 0.15
    morphology: float = 0.10
    source_prior: float = 0.05
    local_image_support: float = 0.05


def score_candidates(
    features: pd.DataFrame,
    *,
    weights: SoftHybridWeights = SoftHybridWeights(),
) -> pd.DataFrame:
    scored = features.copy()
    scored["soft_fusion_support"] = _fill_unit_interval(
        scored.get("fusion_support_rank_in_image"), default=0.5
    )
    scored["soft_qa"] = _fill_unit_interval(
        scored.get(
            "pairwise_cluster_rank",
            scored.get(
                "candidate_ranker_rank_in_cluster",
                scored.get(
                    "candidate_ranker_rank_in_image", scored.get("qa_rank_in_image")
                ),
            ),
        ),
        default=0.5,
    )
    scored["soft_temporal"] = _temporal_score(scored)
    scored["soft_morphology"] = _morphology_score(scored)
    scored["soft_source_prior"] = _fill_unit_interval(
        scored.get("source_train_mean_iou"), default=scored.get("dataset_train_mean_iou")
    )
    if "local_image_support_score" in scored.columns:
        scored["soft_local_image_support"] = _fill_unit_interval(
            scored["local_image_support_score"], default=0.5
        )
    else:
        scored["soft_local_image_support"] = 0.5

    scored["soft_hybrid_score"] = (
        weights.fusion_support * scored["soft_fusion_support"]
        + weights.qa * scored["soft_qa"]
        + weights.temporal * scored["soft_temporal"]
        + weights.morphology * scored["soft_morphology"]
        + weights.source_prior * scored["soft_source_prior"]
        + weights.local_image_support * scored["soft_local_image_support"]
    )
    return scored


def add_local_image_support_score(
    features: pd.DataFrame, *, mode: str = "average"
) -> pd.DataFrame:
    augmented = features.copy()
    full_boundary = _numeric_feature(
        augmented, "ranker_cluster_full_boundary_contrast_rank"
    )
    rect_boundary = _numeric_feature(
        augmented, "ranker_cluster_rect_boundary_contrast_rank"
    )
    full_contrast = _numeric_feature(
        augmented, "ranker_cluster_full_raw_contrast_norm_rank"
    )
    rect_contrast = _numeric_feature(
        augmented, "ranker_cluster_rect_raw_contrast_norm_rank"
    )

    if mode == "average":
        support = pd.concat(
            [full_boundary, rect_boundary, full_contrast, rect_contrast],
            axis=1,
        ).mean(axis=1, skipna=True)
    elif mode == "boundary_mean":
        support = pd.concat([full_boundary, rect_boundary], axis=1).mean(
            axis=1, skipna=True
        )
    elif mode == "boundary_min":
        support = pd.concat([full_boundary, rect_boundary], axis=1).min(
            axis=1, skipna=True
        )
    elif mode == "full_boundary":
        support = full_boundary
    elif mode == "rect_boundary":
        support = rect_boundary
    else:
        raise ValueError(
            "Unsupported local image support mode: "
            f"{mode!r}. Expected one of average, boundary_mean, boundary_min, "
            "full_boundary, rect_boundary."
        )

    if support.isna().all():
        augmented["local_image_support_score"] = 0.5
        return augmented
    augmented["local_image_support_score"] = support.fillna(0.5).clip(0.0, 1.0)
    return augmented


def select_candidate_set(
    scored: pd.DataFrame,
    *,
    max_bbox_iou: float | None = None,
    min_center_distance: float | None = 4.0,
    min_score: float | None = None,
) -> pd.DataFrame:
    required = {
        "image_id",
        "soft_hybrid_score",
        "crop_y_start",
        "crop_y_end",
        "crop_x_start",
        "crop_x_end",
        "center_y",
        "center_x",
    }
    missing = sorted(required.difference(scored.columns))
    if missing:
        raise ValueError(f"Scored candidates are missing columns: {missing}")

    selected_groups = []
    for _, group in scored.groupby("image_id", dropna=False, sort=False):
        ordered = group.sort_values(
            ["soft_hybrid_score", "fusion_support", "qa_score"],
            ascending=[False, False, False],
            na_position="last",
        )
        accepted: list[pd.Series] = []
        accepted_boxes: list[np.ndarray] = []
        accepted_centers: list[np.ndarray] = []
        for _, row in ordered.iterrows():
            score = row["soft_hybrid_score"]
            if min_score is not None and (pd.isna(score) or float(score) < min_score):
                continue

            box = _row_box(row)
            if box is None:
                continue
            center = _row_center(row)
            if center is None:
                continue
            if max_bbox_iou is not None and any(
                _bbox_iou(box, other) > max_bbox_iou for other in accepted_boxes
            ):
                continue
            if min_center_distance is not None and any(
                _center_distance(center, other) < min_center_distance
                for other in accepted_centers
            ):
                continue
            accepted.append(row)
            accepted_boxes.append(box)
            accepted_centers.append(center)

        if accepted:
            selected_groups.append(pd.DataFrame(accepted))

    if not selected_groups:
        return scored.iloc[0:0].copy()

    selected = pd.concat(selected_groups, ignore_index=True)
    selected["selected_by_soft_hybrid"] = True
    return selected


def run_soft_hybrid_selection(
    features: pd.DataFrame,
    *,
    max_bbox_iou: float | None = None,
    min_center_distance: float | None = 4.0,
    min_score: float | None = None,
    weights: SoftHybridWeights = SoftHybridWeights(),
) -> pd.DataFrame:
    scored = score_candidates(features, weights=weights)
    return select_candidate_set(
        scored,
        max_bbox_iou=max_bbox_iou,
        min_center_distance=min_center_distance,
        min_score=min_score,
    )


def _fill_unit_interval(value: object, *, default: object) -> pd.Series:
    if isinstance(value, pd.Series):
        series = pd.to_numeric(value, errors="coerce")
    else:
        series = pd.Series(np.nan)

    if isinstance(default, pd.Series):
        fallback = pd.to_numeric(default, errors="coerce")
    else:
        fallback = default

    series = series.fillna(fallback)
    return series.fillna(0.5).clip(lower=0.0, upper=1.0)


def _temporal_score(features: pd.DataFrame) -> pd.Series:
    source_support = _numeric_feature(
        features, "source_temporal_support_score"
    )
    any_support = _numeric_feature(features, "temporal_support_score")
    support = source_support.fillna(any_support)

    source_distance = _numeric_feature(features, "source_temporal_min_center_dist_norm")
    any_distance = _numeric_feature(features, "temporal_min_center_dist_norm")
    distance = source_distance.fillna(any_distance)
    score = support.fillna(1.0 / (1.0 + distance))
    return score.fillna(0.5).clip(lower=0.0, upper=1.0)


def _numeric_feature(features: pd.DataFrame, column: str) -> pd.Series:
    if column not in features.columns:
        return pd.Series(np.nan, index=features.index, dtype=float)
    return pd.to_numeric(features[column], errors="coerce")


def _morphology_score(features: pd.DataFrame) -> pd.Series:
    area = pd.to_numeric(features.get("crop_area"), errors="coerce")
    image_median = area.groupby(features["image_id"], dropna=False).transform("median")
    ratio = area / image_median.replace(0, np.nan)
    score = 1.0 - np.abs(np.log(ratio.replace(0, np.nan)))
    return score.fillna(0.5).clip(lower=0.0, upper=1.0)


def _row_box(row: pd.Series) -> np.ndarray | None:
    values = [
        row.get("crop_y_start"),
        row.get("crop_x_start"),
        row.get("crop_y_end"),
        row.get("crop_x_end"),
    ]
    if any(pd.isna(value) for value in values):
        return None
    y0, x0, y1, x1 = [float(value) for value in values]
    if y1 <= y0 or x1 <= x0:
        return None
    return np.array([y0, x0, y1, x1], dtype=float)


def _row_center(row: pd.Series) -> np.ndarray | None:
    values = [row.get("center_y"), row.get("center_x")]
    if any(pd.isna(value) for value in values):
        return None
    return np.array([float(values[0]), float(values[1])], dtype=float)


def _center_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    y0 = max(a[0], b[0])
    x0 = max(a[1], b[1])
    y1 = min(a[2], b[2])
    x1 = min(a[3], b[3])
    intersection = max(0.0, y1 - y0) * max(0.0, x1 - x0)
    if intersection <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score candidate features and select a soft-hybrid candidate set."
    )
    parser.add_argument("features", type=Path, help="Candidate feature CSV/Parquet.")
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--max-bbox-iou", type=float, default=None)
    parser.add_argument("--min-center-distance", type=float, default=4.0)
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    if args.features.suffix == ".parquet":
        features = pd.read_parquet(args.features)
    else:
        features = pd.read_csv(args.features)

    selected = run_soft_hybrid_selection(
        features,
        max_bbox_iou=args.max_bbox_iou,
        min_center_distance=args.min_center_distance,
        min_score=args.min_score,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        selected.to_parquet(args.output, index=False)
    else:
        selected.to_csv(args.output, index=False)
    print(f"Wrote {len(selected)} selected candidate rows to {args.output}")


if __name__ == "__main__":
    main()
