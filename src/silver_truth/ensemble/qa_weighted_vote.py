from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import tifffile

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    SILVER_TRUTH_COLUMN,
)
from silver_truth.evaluation.ranker_pruned_fusion import infer_paper_input_metadata
from silver_truth.fusion.crops_experiment import (
    _compute_metrics,
    _gt_mask_from_metadata,
    _normalize_campaign,
    _normalize_label,
    _resolve_stacked_path,
)


DEFAULT_WEIGHT_COLUMNS = (
    "pairwise_cluster_score",
    "soft_hybrid_score",
    "candidate_ranker_score",
    "qa_score",
)

REQUIRED_COLUMNS = (
    "campaign_number",
    "original_image_key",
    "label",
    "stacked_path",
)


def run_qa_weighted_object_vote(
    *,
    candidate_parquet: Path,
    output_dir: Path,
    feature_parquet: Path | None = None,
    weights_column: str | None = None,
    threshold: float = 0.50,
    min_weight: float = 1e-6,
    boundary_support_column: str | None = None,
    boundary_floor: float = 0.25,
    boundary_floor_column: str | None = None,
    boundary_iterations: int = 1,
    top_k_per_group: int | None = None,
    method_key: str = "ensemble_qa_weighted_object_vote",
) -> dict[str, str]:
    """Fuse candidate object masks with an internal QA-weighted pixel vote.

    This intentionally does not call the external label-fusion Java backend.
    Rows are grouped by logical object crop, each candidate mask contributes a
    per-row weight, and the final object mask is the weighted foreground vote.
    """

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in the interval (0, 1].")

    output_dir.mkdir(parents=True, exist_ok=True)
    fused_dir = output_dir / "fused"
    gt_dir = output_dir / "gt"
    fused_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(candidate_parquet)
    _validate_columns(candidates)
    if feature_parquet is not None:
        candidates = _merge_feature_table(candidates, pd.read_parquet(feature_parquet))
    if "competitor" in candidates.columns:
        candidates = candidates[
            ~candidates["competitor"].astype(str).eq(SILVER_TRUTH_COLUMN)
        ].copy()

    resolved_weight_column = _resolve_weight_column(candidates, weights_column)
    resolved_boundary_column = _resolve_boundary_support_column(
        candidates, boundary_support_column
    )
    resolved_boundary_floor_column = _resolve_boundary_floor_column(
        candidates, boundary_floor_column
    )
    candidates = _apply_top_k_per_group(
        candidates,
        weights_column=resolved_weight_column,
        top_k_per_group=top_k_per_group,
    )
    method_key = _resolve_method_key(
        method_key,
        resolved_weight_column,
        boundary_support_column=resolved_boundary_column,
        top_k_per_group=top_k_per_group,
    )
    eval_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    groups = candidates.groupby(_group_keys(candidates), dropna=False, sort=True)

    for index, (key, group) in enumerate(groups):
        group_key = _normalize_group_key(key)
        fused_path = fused_dir / f"fused_{index:04d}.tif"
        gt_path = gt_dir / f"mask{index:04d}.tif"

        result = _fuse_group(
            group,
            qa_parquet=candidate_parquet,
            weights_column=resolved_weight_column,
            threshold=threshold,
            min_weight=min_weight,
            boundary_support_column=resolved_boundary_column,
            boundary_floor=boundary_floor,
            boundary_floor_column=resolved_boundary_floor_column,
            boundary_iterations=boundary_iterations,
        )
        if result is None:
            skipped_rows.append(
                {
                    "campaign_number": group_key[0],
                    "original_image_key": group_key[1],
                    "label": group_key[2],
                    "reason": "no_readable_candidate_masks",
                }
            )
            continue

        fused_mask, gt_mask, weight_sum, candidates_used = result
        tifffile.imwrite(fused_path, fused_mask.astype(np.uint8))
        tifffile.imwrite(gt_path, gt_mask.astype(np.uint8))
        iou, f1 = _compute_metrics(fused_path, gt_path)

        eval_rows.append(
            {
                "method_key": method_key,
                "campaign_number": group_key[0],
                "original_image_key": group_key[1],
                "label": group_key[2],
                "split": _first_non_null(group.get("split")),
                "fused_path": str(fused_path),
                "gt_path": str(gt_path),
                "jaccard": iou,
                "f1": f1,
                "candidates_used": candidates_used,
                "candidate_rows": int(len(group)),
                "weight_sum": weight_sum,
                "weights_column": resolved_weight_column,
                "boundary_support_column": resolved_boundary_column or "",
                "boundary_floor": boundary_floor
                if resolved_boundary_column is not None
                else np.nan,
                "boundary_floor_column": resolved_boundary_floor_column or "",
                "boundary_iterations": boundary_iterations
                if resolved_boundary_column is not None
                else np.nan,
                "top_k_per_group": top_k_per_group
                if top_k_per_group is not None
                else np.nan,
                "threshold": threshold,
            }
        )

    eval_df = pd.DataFrame.from_records(eval_rows)
    skipped_df = pd.DataFrame.from_records(skipped_rows)
    summary_df = summarize_weighted_vote_eval(
        eval_df,
        candidate_parquet=candidate_parquet,
        method_key=method_key,
        weights_column=resolved_weight_column,
        boundary_support_column=resolved_boundary_column,
        boundary_floor=boundary_floor,
        boundary_floor_column=resolved_boundary_floor_column,
        boundary_iterations=boundary_iterations,
        top_k_per_group=top_k_per_group,
        threshold=threshold,
    )
    comparison_df = build_weighted_vote_comparison_rows(summary_df, candidate_parquet)

    eval_path = output_dir / "weighted_object_vote_cell_eval.csv"
    summary_path = output_dir / "weighted_object_vote_summary.csv"
    comparison_path = output_dir / "weighted_object_vote_comparison.csv"
    skipped_path = output_dir / "weighted_object_vote_skipped.csv"
    eval_df.to_csv(eval_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    return {
        "eval_path": str(eval_path),
        "summary_path": str(summary_path),
        "comparison_path": str(comparison_path),
        "skipped_path": str(skipped_path),
    }


def summarize_weighted_vote_eval(
    eval_df: pd.DataFrame,
    *,
    candidate_parquet: Path,
    method_key: str,
    weights_column: str,
    boundary_support_column: str | None = None,
    boundary_floor: float = 0.25,
    boundary_floor_column: str | None = None,
    boundary_iterations: int = 1,
    top_k_per_group: int | None = None,
    threshold: float,
) -> pd.DataFrame:
    metadata = infer_paper_input_metadata_from_pruned_path(candidate_parquet)
    records: list[dict[str, Any]] = []
    split_values: Sequence[str | None]
    if "split" in eval_df.columns and eval_df["split"].notna().any():
        split_values = [*sorted(str(v) for v in eval_df["split"].dropna().unique()), None]
    else:
        split_values = [None]

    for split in split_values:
        subset = eval_df if split is None else eval_df[eval_df["split"].astype(str).eq(split)]
        records.append(
            {
                **metadata,
                "method_key": method_key,
                "method_label": _method_label(
                    weights_column,
                    boundary_support_column=boundary_support_column,
                    top_k_per_group=top_k_per_group,
                ),
                "row_family": _row_family(
                    weights_column,
                    boundary_support_column=boundary_support_column,
                ),
                "split": "overall" if split is None else split,
                "mean_iou": _mean(subset, "jaccard"),
                "mean_f1": _mean(subset, "f1"),
                "n_cells": int(len(subset)),
                "mean_candidates_used": _mean(subset, "candidates_used"),
                "weights_column": weights_column,
                "boundary_support_column": boundary_support_column or "",
                "boundary_floor": boundary_floor
                if boundary_support_column is not None
                else np.nan,
                "boundary_floor_column": boundary_floor_column or "",
                "boundary_iterations": boundary_iterations
                if boundary_support_column is not None
                else np.nan,
                "top_k_per_group": top_k_per_group
                if top_k_per_group is not None
                else np.nan,
                "threshold": threshold,
            }
        )
    return pd.DataFrame.from_records(records)


def build_weighted_vote_comparison_rows(
    summary_df: pd.DataFrame,
    candidate_parquet: Path,
) -> pd.DataFrame:
    metadata = infer_paper_input_metadata_from_pruned_path(candidate_parquet)
    test = summary_df[summary_df["split"].eq("test")]
    source = test if not test.empty else summary_df[summary_df["split"].eq("overall")]
    if source.empty:
        return pd.DataFrame()
    row = source.iloc[0].to_dict()
    return pd.DataFrame(
        [
            {
                **metadata,
                "row_family": _row_family(
                    str(row["weights_column"]),
                    boundary_support_column=_string_or_none(
                        row.get("boundary_support_column")
                    ),
                ),
                "method_key": row["method_key"],
                "method_label": row["method_label"],
                "mean_iou": row["mean_iou"],
                "mean_f1": row["mean_f1"],
                "selection_split": row["split"],
                "fusion_backend": "",
                "weights_column": row["weights_column"],
                "boundary_support_column": row.get("boundary_support_column", ""),
                "boundary_floor": row.get("boundary_floor", np.nan),
                "boundary_floor_column": row.get("boundary_floor_column", ""),
                "boundary_iterations": row.get("boundary_iterations", np.nan),
                "top_k_per_group": row.get("top_k_per_group", np.nan),
                "threshold": row["threshold"],
                "n_cells": row["n_cells"],
                "mean_candidates_used": row["mean_candidates_used"],
            }
        ]
    )


def infer_paper_input_metadata_from_pruned_path(path: Path) -> dict[str, str]:
    metadata = infer_paper_input_metadata(path)
    if metadata["dataset"]:
        return metadata
    parts = path.parts
    for index, part in enumerate(parts):
        if part == "ranker_pruned_fusion_probe" and len(parts) > index + 4:
            return {
                "dataset": parts[index + 1],
                "crop_tag": parts[index + 2],
                "variant": parts[index + 3],
                "fold": next((p for p in parts[index + 4 :] if p.startswith("fold-")), ""),
                "input_path": str(path),
            }
    return metadata


def _fuse_group(
    group: pd.DataFrame,
    *,
    qa_parquet: Path,
    weights_column: str,
    threshold: float,
    min_weight: float,
    boundary_support_column: str | None,
    boundary_floor: float,
    boundary_floor_column: str | None,
    boundary_iterations: int,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    masks: list[np.ndarray] = []
    weights: list[float] = []
    support_values: list[float] = []
    gt_mask: np.ndarray | None = None
    reference_shape: tuple[int, int] | None = None

    for _, row in group.iterrows():
        stack_path = _resolve_stacked_path(row.get("stacked_path"), qa_parquet)
        if stack_path is None or not stack_path.exists():
            continue
        try:
            stack = tifffile.imread(stack_path)
        except Exception:
            continue
        if stack.ndim != 3 or stack.shape[0] < 2:
            continue

        mask = (stack[1] > 0).astype(np.float32)
        if reference_shape is None:
            reference_shape = mask.shape
        if mask.shape != reference_shape:
            mask = _crop_to_shape(mask, reference_shape)
        masks.append(mask)
        weights.append(_row_weight(row, weights_column, min_weight=min_weight))
        support_values.append(
            _row_support(
                row,
                boundary_support_column,
                default=1.0,
                floor=_row_boundary_floor(
                    row,
                    boundary_floor_column,
                    default=boundary_floor,
                ),
            )
        )

        if gt_mask is None:
            if stack.shape[0] >= 3:
                gt_mask = (stack[2] > 0).astype(np.uint8)
            else:
                metadata = _metadata_from_row(row, qa_parquet)
                gt_mask = _gt_mask_from_metadata(
                    metadata,
                    _normalize_label(row.get("label")),
                    reference_shape,
                )

    if not masks or reference_shape is None:
        return None
    if gt_mask is None:
        gt_mask = np.zeros(reference_shape, dtype=np.uint8)
    if gt_mask.shape != reference_shape:
        gt_mask = _crop_to_shape(gt_mask, reference_shape).astype(np.uint8)

    weights_array = np.asarray(weights, dtype=np.float32)
    weight_sum = float(weights_array.sum())
    if weight_sum <= 0:
        weights_array = np.ones(len(masks), dtype=np.float32)
        weight_sum = float(weights_array.sum())
    if boundary_support_column is None:
        vote = np.tensordot(weights_array, np.stack(masks, axis=0), axes=(0, 0))
    else:
        vote = _boundary_aware_vote(
            masks=masks,
            weights=weights_array,
            support_values=np.asarray(support_values, dtype=np.float32),
            boundary_iterations=boundary_iterations,
        )
    fused = (vote / weight_sum >= threshold).astype(np.uint8)
    return fused, gt_mask.astype(np.uint8), weight_sum, len(masks)


def _metadata_from_row(row: pd.Series, qa_parquet: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if pd.notna(row.get("gt_image")):
        metadata["gt_image_path"] = _resolve_stacked_path(row.get("gt_image"), qa_parquet)
    for column in ("crop_y_start", "crop_y_end", "crop_x_start", "crop_x_end"):
        if pd.notna(row.get(column)):
            metadata[column] = int(row[column])
    return metadata


def _crop_to_shape(source: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    cropped = np.zeros(target_shape, dtype=source.dtype)
    y_size = min(target_shape[0], source.shape[0])
    x_size = min(target_shape[1], source.shape[1])
    cropped[:y_size, :x_size] = source[:y_size, :x_size]
    return cropped


def _row_weight(row: pd.Series, weights_column: str, *, min_weight: float) -> float:
    value = pd.to_numeric(pd.Series([row.get(weights_column)]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value):
        return 1.0
    return max(float(value), min_weight)


def _row_support(
    row: pd.Series,
    support_column: str | None,
    *,
    default: float,
    floor: float,
) -> float:
    if support_column is None:
        return default
    value = pd.to_numeric(pd.Series([row.get(support_column)]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value):
        value = default
    return float(np.clip(value, floor, 1.0))


def _row_boundary_floor(
    row: pd.Series,
    floor_column: str | None,
    *,
    default: float,
) -> float:
    if floor_column is None:
        return float(np.clip(default, 0.0, 1.0))
    value = pd.to_numeric(pd.Series([row.get(floor_column)]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value):
        value = default
    return float(np.clip(value, 0.0, 1.0))


def _boundary_aware_vote(
    *,
    masks: list[np.ndarray],
    weights: np.ndarray,
    support_values: np.ndarray,
    boundary_iterations: int,
) -> np.ndarray:
    vote = np.zeros_like(masks[0], dtype=np.float32)
    for mask, weight, support in zip(masks, weights, support_values, strict=True):
        boundary = _mask_boundary(mask > 0, iterations=boundary_iterations)
        pixel_weight = np.where(boundary, float(weight * support), float(weight))
        vote += mask * pixel_weight.astype(np.float32)
    return vote


def _mask_boundary(mask: np.ndarray, *, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return np.zeros(mask.shape, dtype=bool)
    eroded = mask.astype(bool)
    for _ in range(iterations):
        eroded = _binary_erosion_3x3(eroded)
    return mask.astype(bool) & ~eroded


def _binary_erosion_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    eroded = np.ones(mask.shape, dtype=bool)
    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    return eroded


def _resolve_weight_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested in {"_uniform_weight", "uniform"}:
            return "_uniform_weight"
        if requested not in df.columns:
            raise ValueError(f"Requested weights column {requested!r} not found.")
        return requested
    for column in DEFAULT_WEIGHT_COLUMNS:
        if column in df.columns:
            return column
    return "_uniform_weight"


def _resolve_boundary_support_column(
    df: pd.DataFrame, requested: str | None
) -> str | None:
    if requested is None:
        return None
    if requested not in df.columns:
        raise ValueError(f"Requested boundary support column {requested!r} not found.")
    return requested


def _resolve_boundary_floor_column(
    df: pd.DataFrame,
    requested: str | None,
) -> str | None:
    if requested is None:
        return None
    if requested not in df.columns:
        raise ValueError(f"Requested boundary floor column {requested!r} not found.")
    return requested


def _resolve_method_key(
    method_key: str,
    weights_column: str,
    *,
    boundary_support_column: str | None,
    top_k_per_group: int | None,
) -> str:
    if method_key != "ensemble_qa_weighted_object_vote":
        return method_key
    if boundary_support_column is not None:
        if top_k_per_group is not None:
            return "ensemble_qa_boundary_topk_object_vote"
        return "ensemble_qa_boundary_object_vote"
    if weights_column == "_uniform_weight":
        if top_k_per_group is not None:
            return "ensemble_only_uniform_topk_object_vote"
        return "ensemble_only_uniform_object_vote"
    if top_k_per_group is not None:
        return "ensemble_qa_weighted_topk_object_vote"
    return method_key


def _row_family(
    weights_column: str,
    *,
    boundary_support_column: str | None = None,
) -> str:
    if boundary_support_column is not None:
        return "ensemble_qa"
    if weights_column == "_uniform_weight":
        return "ensemble_only"
    return "ensemble_qa"


def _method_label(
    weights_column: str,
    *,
    boundary_support_column: str | None = None,
    top_k_per_group: int | None = None,
) -> str:
    topk = f" top-{top_k_per_group}" if top_k_per_group is not None else ""
    if boundary_support_column is not None:
        return f"Boundary-aware{topk} object vote"
    if weights_column == "_uniform_weight":
        return f"Uniform{topk} object vote"
    return f"QA-weighted{topk} object vote"


def _merge_feature_table(candidates: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    left_key = "cell_id" if "cell_id" in candidates.columns else "candidate_id"
    right_key = "candidate_id" if "candidate_id" in features.columns else "cell_id"
    if left_key not in candidates.columns or right_key not in features.columns:
        raise ValueError(
            "Cannot merge feature table: expected candidate cell_id/candidate_id "
            "and feature candidate_id/cell_id columns."
        )
    feature_columns = [
        column
        for column in features.columns
        if column == right_key or column not in candidates.columns
    ]
    feature_subset = features[feature_columns].drop_duplicates(
        subset=[right_key], keep="first"
    )
    merged = candidates.merge(
        feature_subset,
        how="left",
        left_on=left_key,
        right_on=right_key,
    )
    if right_key != left_key and right_key in merged.columns:
        merged = merged.drop(columns=[right_key])
    return merged


def _apply_top_k_per_group(
    candidates: pd.DataFrame,
    *,
    weights_column: str,
    top_k_per_group: int | None,
) -> pd.DataFrame:
    if top_k_per_group is None:
        return candidates
    if top_k_per_group <= 0:
        raise ValueError("top_k_per_group must be positive when provided.")

    if weights_column == "_uniform_weight":
        score = pd.Series(1.0, index=candidates.index)
    else:
        score = pd.to_numeric(candidates[weights_column], errors="coerce")
    group_columns = (
        ["ranker_cluster_id"]
        if "ranker_cluster_id" in candidates.columns
        else _group_keys(candidates)
    )

    ranked = candidates.copy()
    ranked["_object_vote_topk_score"] = score.fillna(float("-inf"))
    ranked["_object_vote_original_order"] = np.arange(len(ranked))
    ranked = ranked.sort_values(
        [*group_columns, "_object_vote_topk_score", "_object_vote_original_order"],
        ascending=[*[True for _ in group_columns], False, True],
        kind="mergesort",
    )
    keep = ranked.groupby(group_columns, dropna=False, sort=False).cumcount() < int(
        top_k_per_group
    )
    selected = ranked.loc[keep].sort_values("_object_vote_original_order")
    return selected.drop(
        columns=["_object_vote_topk_score", "_object_vote_original_order"]
    ).reset_index(drop=True)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
    if missing:
        raise ValueError(f"Candidate parquet is missing columns: {', '.join(missing)}")


def _group_keys(df: pd.DataFrame) -> list[str]:
    keys = ["campaign_number", "original_image_key", "label"]
    return [key for key in keys if key in df.columns]


def _normalize_group_key(key: Any) -> tuple[str, str, Any]:
    if not isinstance(key, tuple):
        key = (key,)
    campaign = _normalize_campaign(key[0]) if len(key) > 0 else "unknown"
    image = str(key[1]) if len(key) > 1 else "unknown"
    label = _normalize_label(key[2]) if len(key) > 2 else "unknown"
    return campaign, image, label


def _first_non_null(values: pd.Series | None) -> Any:
    if values is None:
        return np.nan
    for value in values:
        if pd.notna(value):
            return value
    return np.nan


def _string_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").mean())
