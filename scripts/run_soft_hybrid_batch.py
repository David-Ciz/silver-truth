#!/usr/bin/env python3
"""Run soft-hybrid export, selection, reconstruction, and summary in batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from silver_truth.evaluation.candidate_features import write_candidate_feature_table
from silver_truth.evaluation.candidate_ranker import (
    CandidateRankerConfig,
    CROP_GEOMETRY_RANKER_FEATURE_COLUMNS,
    DEFAULT_RANKER_FEATURE_COLUMNS,
    FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS,
    FULL_IMAGE_MASK_RANKER_FEATURE_COLUMNS,
    IMAGE_MASK_RANKER_FEATURE_COLUMNS,
    MULTISCALE_RECT_IMAGE_MASK_RANKER_FEATURE_COLUMNS,
    RECT_IMAGE_MASK_RANKER_FEATURE_COLUMNS,
    TEMPORAL_SUPPORT_RANKER_FEATURE_COLUMNS,
    score_with_candidate_ranker,
    train_candidate_ranker,
)
from silver_truth.evaluation.soft_hybrid import (
    add_local_image_support_score,
    run_soft_hybrid_selection,
)
from silver_truth.evaluation.soft_hybrid_reconstruction import (
    reconstruct_selected_soft_hybrid_candidates,
)


def discover_paper_ready_inputs(roots: list[Path]) -> list[Path]:
    inputs: list[Path] = []
    for root in roots:
        inputs.extend(root.glob("**/paper_inputs/**/fold-*paper_ready.parquet"))
    return sorted(set(inputs))


def output_subdir(input_path: Path, output_root: Path, method_name: str) -> Path:
    parts = input_path.parts
    if "paper_inputs" not in parts:
        return output_root / input_path.stem / method_name
    index = parts.index("paper_inputs")
    if len(parts) <= index + 4:
        return output_root / input_path.stem / method_name
    dataset = parts[index + 1]
    crop_tag = parts[index + 2]
    variant = parts[index + 3]
    fold = input_path.stem.split("_", 1)[0]
    return output_root / dataset / crop_tag / variant / fold / method_name


def summarize_eval(
    eval_df: pd.DataFrame, *, input_path: Path, output_dir: Path
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split, split_df in eval_df.groupby("split", dropna=False):
        records.append(
            {
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "split": split,
                "n_images": int(len(split_df)),
                "mean_iou": _metric_mean(split_df, "iou"),
                "mean_f1": _metric_mean(split_df, "f1"),
                "mean_cells_considered": float(
                    pd.to_numeric(
                        split_df.get("cells_considered"), errors="coerce"
                    ).mean()
                ),
                "mean_cells_placed": float(
                    pd.to_numeric(split_df.get("cells_placed"), errors="coerce").mean()
                ),
                "mean_labels_scored": float(
                    pd.to_numeric(split_df.get("labels_scored"), errors="coerce").mean()
                ),
            }
        )
    return records


def _metric_mean(table: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(table[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return float("nan")

    if "labels_scored" not in table.columns:
        return float(values[valid].mean())

    weights = pd.to_numeric(table["labels_scored"], errors="coerce")
    weight_valid = valid & weights.notna() & (weights > 0)
    if not weight_valid.any():
        return float(values[valid].mean())
    return float(
        (values[weight_valid] * weights[weight_valid]).sum()
        / weights[weight_valid].sum()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run soft-hybrid candidate selection for paper-ready inputs."
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        help="Root to search for paper_inputs. Can be repeated.",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Explicit fold-*paper_ready.parquet input. Can be repeated.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-bbox-iou", type=float, default=None)
    parser.add_argument("--min-center-distance", type=float, default=4.0)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--use-candidate-ranker",
        action="store_true",
        help="Train a per-input calibrated candidate ranker before soft-hybrid scoring.",
    )
    parser.add_argument(
        "--use-source-identity",
        action="store_true",
        help="Include canonical candidate_source as a categorical ranker feature.",
    )
    parser.add_argument(
        "--ranker-train-split",
        action="append",
        default=None,
        help="Split used to train the candidate ranker. Can be repeated.",
    )
    parser.add_argument(
        "--ranker-calibration-split",
        action="append",
        default=None,
        help="Split used to calibrate the candidate ranker. Can be repeated.",
    )
    parser.add_argument(
        "--use-source-cluster-quality-prior",
        action="store_true",
        help="Include source-level local-cluster regret quality prior.",
    )
    parser.add_argument(
        "--include-temporal-support-ranker-features",
        action="store_true",
        help=(
            "Opt in to temporal-support features inside the candidate ranker. "
            "The default keeps the earlier safer feature set."
        ),
    )
    parser.add_argument(
        "--include-image-features",
        action="store_true",
        help=(
            "Read stacked crop TIFFs, add mask/image evidence features, and "
            "include them in the candidate ranker."
        ),
    )
    parser.add_argument(
        "--include-rect-image-features",
        action="store_true",
        help=(
            "Read stacked crop TIFFs, add object-local rectangular mask/image "
            "features, and include them in the candidate ranker."
        ),
    )
    parser.add_argument(
        "--include-multiscale-rect-image-features",
        action="store_true",
        help=(
            "Read stacked crop TIFFs, add tight/far rectangular mask/image "
            "features, and include them in the candidate ranker."
        ),
    )
    parser.add_argument(
        "--include-full-image-features",
        action="store_true",
        help=(
            "Read original images, add object-local mask/image features in true "
            "image coordinates, and include them in the candidate ranker."
        ),
    )
    parser.add_argument(
        "--include-full-image-boundary-noise-features",
        action="store_true",
        help=(
            "Add full-image boundary/noise diagnostics and include them in the "
            "candidate ranker. This also enables base full-image features."
        ),
    )
    parser.add_argument(
        "--include-crop-geometry-features",
        action="store_true",
        help=(
            "Read original image shapes, add crop padding/image-border geometry "
            "features, and include them in the candidate ranker."
        ),
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help=(
            "Output method directory. Defaults to soft_hybrid, or "
            "soft_hybrid_ranked when --use-candidate-ranker is set."
        ),
    )
    parser.add_argument(
        "--use-local-image-support-score",
        action="store_true",
        help=(
            "Use cluster-relative rect/full-image contrast ranks as the "
            "soft-hybrid local_image_support term."
        ),
    )
    parser.add_argument(
        "--local-image-support-mode",
        choices=[
            "average",
            "boundary_mean",
            "boundary_min",
            "full_boundary",
            "rect_boundary",
        ],
        default="average",
        help="Cluster-relative image signal used when --use-local-image-support-score is set.",
    )
    args = parser.parse_args()

    if args.input and not args.root:
        inputs = sorted(set(args.input))
    else:
        roots = args.root or [Path("data"), Path("hpc_results")]
        inputs = sorted(set(args.input + discover_paper_ready_inputs(roots)))
    if not inputs:
        raise FileNotFoundError("No fold-*paper_ready.parquet inputs found.")

    method_name = args.method_name or (
        "soft_hybrid_source_ranked"
        if args.use_candidate_ranker and args.use_source_identity
        else "soft_hybrid_ranked"
        if args.use_candidate_ranker
        else "soft_hybrid"
    )
    ranker_feature_columns = DEFAULT_RANKER_FEATURE_COLUMNS
    if args.include_temporal_support_ranker_features:
        ranker_feature_columns = (
            DEFAULT_RANKER_FEATURE_COLUMNS
            + TEMPORAL_SUPPORT_RANKER_FEATURE_COLUMNS
        )
    if args.include_image_features:
        ranker_feature_columns = (
            ranker_feature_columns + IMAGE_MASK_RANKER_FEATURE_COLUMNS
        )
    if args.include_rect_image_features:
        ranker_feature_columns = (
            ranker_feature_columns + RECT_IMAGE_MASK_RANKER_FEATURE_COLUMNS
        )
    if args.include_multiscale_rect_image_features:
        ranker_feature_columns = (
            ranker_feature_columns
            + MULTISCALE_RECT_IMAGE_MASK_RANKER_FEATURE_COLUMNS
        )
    if args.include_full_image_features or args.include_full_image_boundary_noise_features:
        ranker_feature_columns = (
            ranker_feature_columns + FULL_IMAGE_MASK_RANKER_FEATURE_COLUMNS
        )
    if args.include_full_image_boundary_noise_features:
        ranker_feature_columns = (
            ranker_feature_columns + FULL_IMAGE_BOUNDARY_NOISE_RANKER_FEATURE_COLUMNS
        )
    if args.include_crop_geometry_features:
        ranker_feature_columns = (
            ranker_feature_columns + CROP_GEOMETRY_RANKER_FEATURE_COLUMNS
        )
    summary_records: list[dict[str, object]] = []
    for input_path in inputs:
        fold_output = output_subdir(input_path, args.output_root, method_name)
        fold_output.mkdir(parents=True, exist_ok=True)

        features_path = fold_output / "candidate_features.parquet"
        ranked_features_path = fold_output / "candidate_features_ranked.parquet"
        ranker_model_path = fold_output / "candidate_ranker.joblib"
        selected_path = fold_output / "selected_candidates.parquet"

        features = write_candidate_feature_table(
            [input_path],
            features_path,
            include_image_features=args.include_image_features,
            include_rect_image_features=args.include_rect_image_features,
            include_multiscale_rect_image_features=(
                args.include_multiscale_rect_image_features
            ),
            include_full_image_features=(
                args.include_full_image_features
                or args.include_full_image_boundary_noise_features
            ),
            include_full_image_boundary_noise_features=(
                args.include_full_image_boundary_noise_features
            ),
            include_crop_geometry_features=args.include_crop_geometry_features,
        )
        if args.use_candidate_ranker:
            ranker_payload = train_candidate_ranker(
                features,
                output_model=ranker_model_path,
                config=CandidateRankerConfig(
                    feature_columns=ranker_feature_columns,
                    train_splits=tuple(args.ranker_train_split or ("train",)),
                    calibration_splits=tuple(
                        args.ranker_calibration_split or ("validation",)
                    ),
                    categorical_columns=("candidate_source",)
                    if args.use_source_identity
                    else (),
                    use_source_cluster_quality_prior=(
                        args.use_source_cluster_quality_prior
                    ),
                ),
            )
            features = score_with_candidate_ranker(features, model_payload=ranker_payload)
            if args.use_local_image_support_score:
                features = add_local_image_support_score(
                    features, mode=args.local_image_support_mode
                )
            features.to_parquet(ranked_features_path, index=False)
        elif args.use_local_image_support_score:
            features = add_local_image_support_score(
                features, mode=args.local_image_support_mode
            )
        selected = run_soft_hybrid_selection(
            features,
            max_bbox_iou=args.max_bbox_iou,
            min_center_distance=args.min_center_distance,
            min_score=args.min_score,
        )
        selected.to_parquet(selected_path, index=False)
        eval_df = reconstruct_selected_soft_hybrid_candidates(
            candidate_parquet=input_path,
            selected_candidates_path=selected_path,
            output_dir=fold_output,
        )
        summary_records.extend(
            summarize_eval(eval_df, input_path=input_path, output_dir=fold_output)
        )
        print(
            f"{input_path}: {len(features)} features, {len(selected)} selected, "
            f"{len(eval_df)} eval rows"
        )

    summary = pd.DataFrame.from_records(summary_records)
    summary_path = args.output_root / "soft_hybrid_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
