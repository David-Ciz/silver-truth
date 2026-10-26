#!/usr/bin/env python3
"""Reconstruct QA weighted object votes as labeled full images.

This audit script reads the preserved, pruned candidate tables, recreates the
continuous object vote used by ENSEMBLE, pastes every cell into its original
image coordinates, resolves overlaps by pixel confidence, and reports both
cell weighted and frame weighted IoU/F1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from silver_truth.ensemble.qa_weighted_vote import _mask_boundary
from silver_truth.ensemble.reconstruction import (
    reconstruct_labeled_full_images_from_arrays,
)
from silver_truth.metrics.metrics import calculate_labelwise_scores


GROUP_COLUMNS = ["campaign_number", "original_image_key", "label"]


def rebase_path(value: object, repository_root: Path) -> str:
    text = str(value)
    local_prefix = "/Users/davidciz/Work/silver-truth/"
    if text.startswith(local_prefix):
        return str(repository_root / text.removeprefix(local_prefix))
    path = Path(text)
    if not path.is_absolute():
        return str(repository_root / path)
    return text


def crop_to_shape(source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=source.dtype)
    height = min(shape[0], source.shape[0])
    width = min(shape[1], source.shape[1])
    result[:height, :width] = source[:height, :width]
    return result


def fuse_probability(
    group: pd.DataFrame,
    *,
    weight_column: str,
    support_column: str,
    boundary_floor: float,
) -> np.ndarray:
    masks: list[np.ndarray] = []
    weights: list[float] = []
    supports: list[float] = []
    shape: tuple[int, int] | None = None

    for _, row in group.iterrows():
        stack = tifffile.imread(str(row["stacked_path"]))
        mask = (stack[1] > 0).astype(np.float32)
        if shape is None:
            shape = mask.shape
        elif mask.shape != shape:
            mask = crop_to_shape(mask, shape)
        weight = pd.to_numeric(pd.Series([row.get(weight_column)]), errors="coerce").iloc[0]
        support = pd.to_numeric(pd.Series([row.get(support_column)]), errors="coerce").iloc[0]
        weights.append(1.0 if pd.isna(weight) else max(float(weight), 1e-6))
        supports.append(1.0 if pd.isna(support) else float(np.clip(support, boundary_floor, 1.0)))
        masks.append(mask)

    if not masks or shape is None:
        raise ValueError("Object group contains no readable masks")

    vote = np.zeros(shape, dtype=np.float32)
    for mask, weight, support in zip(masks, weights, supports, strict=True):
        boundary = _mask_boundary(mask > 0, iterations=1)
        pixel_weight = np.where(boundary, weight * support, weight)
        vote += mask * pixel_weight.astype(np.float32)
    return vote / float(np.sum(weights))


def reconstruct_one(
    input_path: Path,
    *,
    repository_root: Path,
    output_root: Path,
    weight_column: str,
    support_column: str,
    boundary_floor: float,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_parquet(input_path).copy()
    for column in ("stacked_path", "gt_image", "original_image_path"):
        if column in candidates.columns:
            candidates[column] = candidates[column].map(
                lambda value: rebase_path(value, repository_root)
            )

    rows: list[pd.Series] = []
    probabilities: list[np.ndarray] = []
    for _, group in candidates.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        probability = fuse_probability(
            group,
            weight_column=weight_column,
            support_column=support_column,
            boundary_floor=boundary_floor,
        )
        rows.append(group.iloc[0].copy())
        probabilities.append(probability)

    metadata = pd.DataFrame(rows).reset_index(drop=True)
    relative = input_path.relative_to(input_path.parents[7])
    fold_output = output_root / relative.parent
    frame_eval = reconstruct_labeled_full_images_from_arrays(
        databank_df=metadata,
        predicted_crops=probabilities,
        output_dir=fold_output / "full_images",
        # The original object vote accepts foreground at vote >= threshold,
        # whereas reconstruction uses a strict greater-than comparison.
        threshold=float(np.nextafter(threshold, -np.inf)),
        priority_columns=None,
    )
    frame_eval.to_csv(fold_output / "fullimage_frame_eval.csv", index=False)

    label_rows: list[dict[str, object]] = []
    for row in frame_eval.to_dict("records"):
        gt = tifffile.imread(row["gt_image"])
        prediction = tifffile.imread(row["reconstructed_path"])
        scores = calculate_labelwise_scores(gt, prediction)
        for label, values in scores.items():
            label_rows.append(
                {
                    "campaign_number": row["campaign_number"],
                    "original_image_key": row["original_image_key"],
                    "split": row["split"],
                    "label": int(label),
                    "iou": values["jaccard"],
                    "f1": values["f1"],
                    "present_in_prediction": bool(np.any(prediction == label)),
                }
            )
    label_eval = pd.DataFrame(label_rows)
    label_eval.to_csv(fold_output / "fullimage_label_eval.csv", index=False)
    return frame_eval, label_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=None, type=Path)
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Discover clipped_score_thr050 prepared tables below this directory.",
    )
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--weight-column", default="_qa_soft_gate_weight")
    parser.add_argument("--support-column", default="local_image_support_score")
    parser.add_argument("--boundary-floor", default=0.75, type=float)
    parser.add_argument("--threshold", default=0.50, type=float)
    args = parser.parse_args()

    input_paths = list(args.input or [])
    if args.input_root is not None:
        input_paths.extend(
            sorted(
                args.input_root.glob(
                    "**/source_balanced_topk/clipped_score_thr050/"
                    "prepared_soft_gate_candidates.parquet"
                )
            )
        )
    if not input_paths:
        parser.error("Provide --input or --input-root")

    summaries: list[dict[str, object]] = []
    for input_path in input_paths:
        frame_eval, label_eval = reconstruct_one(
            input_path,
            repository_root=args.repository_root,
            output_root=args.output_root,
            weight_column=args.weight_column,
            support_column=args.support_column,
            boundary_floor=args.boundary_floor,
            threshold=args.threshold,
        )
        test_frames = frame_eval[frame_eval["split"].astype(str).eq("test")]
        test_labels = label_eval[label_eval["split"].astype(str).eq("test")]
        parts = input_path.parts
        dataset = next(part for part in parts if part in {
            "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa",
            "Fluo-C2DL-MSC", "Fluo-N2DH-GOWT1",
        })
        fold = next(part for part in parts if part.startswith("fold-"))
        summaries.append(
            {
                "dataset": dataset,
                "fold": fold,
                "cell_weighted_iou": test_labels["iou"].mean(),
                "cell_weighted_f1": test_labels["f1"].mean(),
                "frame_weighted_iou": test_frames["iou"].mean(),
                "frame_weighted_f1": test_frames["f1"].mean(),
                "test_labels": len(test_labels),
                "missing_test_labels": int((~test_labels["present_in_prediction"]).sum()),
                "input_path": str(input_path),
            }
        )
        print(dataset, fold, summaries[-1]["cell_weighted_iou"])

    summary = pd.DataFrame(summaries).sort_values(["dataset", "fold"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_root / "fold_summary.csv", index=False)
    dataset_summary = (
        summary.groupby("dataset", as_index=False)
        .agg(
            cell_weighted_iou=("cell_weighted_iou", "mean"),
            cell_weighted_f1=("cell_weighted_f1", "mean"),
            frame_weighted_iou=("frame_weighted_iou", "mean"),
            frame_weighted_f1=("frame_weighted_f1", "mean"),
            folds=("fold", "nunique"),
            test_labels=("test_labels", "sum"),
            missing_test_labels=("missing_test_labels", "sum"),
        )
    )
    dataset_summary.to_csv(args.output_root / "dataset_summary.csv", index=False)
    print(dataset_summary.to_string(index=False))
    print("Dataset mean cell weighted IoU:", dataset_summary["cell_weighted_iou"].mean())
    print("Dataset mean frame weighted IoU:", dataset_summary["frame_weighted_iou"].mean())


if __name__ == "__main__":
    main()
