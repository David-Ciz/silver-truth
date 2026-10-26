#!/usr/bin/env python3
"""Rerun the strict Deep Sets ENSEMBLE and evaluate labeled full images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from reconstruct_object_vote_full_images import (
    GROUP_COLUMNS,
    fuse_probability,
    reconstruct_one,
)
from silver_truth.evaluation.candidate_set_weight import (
    CandidateSetWeightConfig,
    write_leave_target_video_out_weighted_features,
)


WEIGHT_COLUMN = "candidate_set_weight_raw_cluster_softmax_t1p000"


def add_cluster_softmax(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    raw = pd.to_numeric(result["candidate_set_weight_raw_score"], errors="coerce")
    groups = result["ranker_cluster_id"].astype(str)
    centered = raw - raw.groupby(groups, dropna=False).transform("max")
    result[WEIGHT_COLUMN] = np.exp(centered.clip(lower=-50.0, upper=0.0)).clip(
        lower=1e-3
    )
    return result


def crop_iou_summary(frame: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, group in frame.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        probability = fuse_probability(
            group,
            weight_column=WEIGHT_COLUMN,
            support_column="local_image_support_score",
            boundary_floor=0.75,
        )
        stack = tifffile.imread(str(group.iloc[0]["stacked_path"]))
        ground_truth = stack[2] > 0
        prediction = probability >= threshold
        intersection = np.logical_and(prediction, ground_truth).sum()
        union = np.logical_or(prediction, ground_truth).sum()
        denominator = prediction.sum() + ground_truth.sum()
        records.append(
            {
                "split": group.iloc[0]["split"],
                "iou": float(intersection / union) if union else 1.0,
                "f1": float(2 * intersection / denominator) if denominator else 1.0,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    feature_paths = sorted(
        args.candidate_root.glob(
            "**/source_balanced_topk/clipped_score_thr050/"
            "prepared_soft_gate_candidates.parquet"
        )
    )
    if len(feature_paths) != 10:
        raise SystemExit(f"Expected 10 candidate tables, found {len(feature_paths)}")

    scored_root = args.output_root / "scored_features"
    model_root = args.output_root / "models"
    summary = write_leave_target_video_out_weighted_features(
        feature_paths,
        output_root=scored_root,
        relative_root=args.candidate_root,
        output_model_root=model_root,
        config=CandidateSetWeightConfig(
            train_splits=("train",),
            model_type="deep_sets",
            deep_sets_epochs=80,
            deep_sets_device="cpu",
        ),
    )
    summary.to_csv(args.output_root / "training_summary.csv", index=False)

    fold_records: list[dict[str, object]] = []
    for row in summary.to_dict("records"):
        scored_path = Path(str(row["output_path"]))
        frame = add_cluster_softmax(pd.read_parquet(scored_path))
        for column in ("stacked_path", "gt_image", "original_image_path"):
            if column in frame.columns:
                frame[column] = frame[column].map(
                    lambda value: str(args.repository_root / str(value))
                    if not Path(str(value)).is_absolute()
                    else str(value)
                )
        calibrated_path = (
            args.output_root / "calibrated_features" / scored_path.relative_to(scored_root)
        )
        calibrated_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(calibrated_path, index=False)

        crop_eval = crop_iou_summary(frame, threshold=0.48)
        frame_eval, label_eval = reconstruct_one(
            calibrated_path,
            repository_root=args.repository_root,
            output_root=args.output_root / "full_image_evaluation",
            weight_column=WEIGHT_COLUMN,
            support_column="local_image_support_score",
            boundary_floor=0.75,
            threshold=0.48,
        )
        test_crop = crop_eval[crop_eval["split"].astype(str).eq("test")]
        test_frame = frame_eval[frame_eval["split"].astype(str).eq("test")]
        test_label = label_eval[label_eval["split"].astype(str).eq("test")]
        fold_records.append(
            {
                "dataset": row["held_out_dataset"],
                "fold": row["held_out_fold"],
                "crop_iou": test_crop["iou"].mean(),
                "crop_f1": test_crop["f1"].mean(),
                "fullimage_cell_weighted_iou": test_label["iou"].mean(),
                "fullimage_cell_weighted_f1": test_label["f1"].mean(),
                "fullimage_frame_weighted_iou": test_frame["iou"].mean(),
                "test_cells": len(test_label),
                "missing_test_cells": int(
                    (~test_label["present_in_prediction"]).sum()
                ),
            }
        )
        print(
            row["held_out_dataset"],
            row["held_out_fold"],
            fold_records[-1]["crop_iou"],
            fold_records[-1]["fullimage_cell_weighted_iou"],
            flush=True,
        )

    folds = pd.DataFrame(fold_records).sort_values(["dataset", "fold"])
    folds.to_csv(args.output_root / "fold_summary.csv", index=False)
    datasets = folds.groupby("dataset", as_index=False).mean(numeric_only=True)
    datasets.to_csv(args.output_root / "dataset_summary.csv", index=False)
    print(datasets.to_string(index=False))
    print("Crop dataset mean IoU:", datasets["crop_iou"].mean())
    print(
        "Full image cell weighted dataset mean IoU:",
        datasets["fullimage_cell_weighted_iou"].mean(),
    )


if __name__ == "__main__":
    main()
