#!/usr/bin/env python3
"""Evaluate a small grid of soft-hybrid weight presets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from silver_truth.evaluation.candidate_features import build_candidate_feature_table
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
from silver_truth.evaluation.cluster_pairwise_ranker import (
    ClusterPairwiseRankerConfig,
    score_with_cluster_pairwise_ranker,
    train_cluster_pairwise_ranker,
)
from silver_truth.evaluation.soft_hybrid import (
    SoftHybridWeights,
    add_local_image_support_score,
    run_soft_hybrid_selection,
)
from silver_truth.evaluation.soft_hybrid_reconstruction import (
    reconstruct_selected_soft_hybrid_candidates,
)


PRESETS = {
    "balanced": SoftHybridWeights(),
    "fusion_heavy": SoftHybridWeights(
        fusion_support=0.55,
        qa=0.15,
        temporal=0.15,
        morphology=0.05,
        source_prior=0.05,
        local_image_support=0.05,
    ),
    "qa_heavy": SoftHybridWeights(
        fusion_support=0.25,
        qa=0.45,
        temporal=0.15,
        morphology=0.05,
        source_prior=0.05,
        local_image_support=0.05,
    ),
    "temporal_heavy": SoftHybridWeights(
        fusion_support=0.35,
        qa=0.20,
        temporal=0.30,
        morphology=0.05,
        source_prior=0.05,
        local_image_support=0.05,
    ),
    "support_qa": SoftHybridWeights(
        fusion_support=0.50,
        qa=0.35,
        temporal=0.05,
        morphology=0.05,
        source_prior=0.00,
        local_image_support=0.05,
    ),
    "no_qa": SoftHybridWeights(
        fusion_support=0.55,
        qa=0.00,
        temporal=0.20,
        morphology=0.15,
        source_prior=0.05,
        local_image_support=0.05,
    ),
}


def output_subdir(
    input_path: Path, output_root: Path, preset: str, method_prefix: str
) -> Path:
    parts = input_path.parts
    method_name = f"{method_prefix}_{preset}"
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


def feature_output_path(input_path: Path, output_root: Path) -> Path:
    parts = input_path.parts
    if "paper_inputs" not in parts:
        return output_root / "features" / f"{input_path.stem}.parquet"
    index = parts.index("paper_inputs")
    if len(parts) <= index + 4:
        return output_root / "features" / f"{input_path.stem}.parquet"
    dataset = parts[index + 1]
    crop_tag = parts[index + 2]
    variant = parts[index + 3]
    return output_root / "features" / dataset / crop_tag / variant / f"{input_path.stem}.parquet"


def summarize_eval(
    eval_df: pd.DataFrame,
    *,
    input_path: Path,
    output_dir: Path,
    preset: str,
    weights: SoftHybridWeights,
    use_candidate_ranker: bool,
    use_source_identity: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    weight_dict = asdict(weights)
    for split, split_df in eval_df.groupby("split", dropna=False):
        record: dict[str, object] = {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "preset": preset,
            "use_candidate_ranker": use_candidate_ranker,
            "use_source_identity": use_source_identity,
            "split": split,
            "n_images": int(len(split_df)),
            "mean_iou": _metric_mean(split_df, "iou"),
            "mean_f1": _metric_mean(split_df, "f1"),
            "mean_cells_placed": float(
                pd.to_numeric(split_df.get("cells_placed"), errors="coerce").mean()
            ),
            "mean_labels_scored": float(
                pd.to_numeric(split_df.get("labels_scored"), errors="coerce").mean()
            ),
        }
        record.update({f"weight_{key}": value for key, value in weight_dict.items()})
        records.append(record)
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
        description="Tune soft-hybrid weight presets on paper-ready candidates."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="fold-*paper_ready.parquet input. Can be repeated.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS),
        help="Preset to evaluate. Can be repeated. Defaults to all presets.",
    )
    parser.add_argument("--max-bbox-iou", type=float, default=None)
    parser.add_argument("--min-center-distance", type=float, default=4.0)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--use-candidate-ranker",
        action="store_true",
        help="Train a per-input calibrated candidate ranker before grid scoring.",
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
        "--candidate-ranker-model-type",
        choices=("ridge", "mlp"),
        default="ridge",
        help="Regressor used by --use-candidate-ranker before calibration.",
    )
    parser.add_argument(
        "--candidate-ranker-mlp-hidden-layer-size",
        action="append",
        type=int,
        help="Hidden layer size for MLP candidate ranker. Can be repeated.",
    )
    parser.add_argument("--candidate-ranker-mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--candidate-ranker-mlp-max-iter", type=int, default=500)
    parser.add_argument(
        "--use-pairwise-cluster-ranker",
        action="store_true",
        help="Train a pairwise local-cluster ranker before grid scoring.",
    )
    parser.add_argument(
        "--include-temporal-support-ranker-features",
        action="store_true",
        help=(
            "Opt in to temporal-support features inside candidate/pairwise "
            "rankers. Default ranker features keep the earlier safer behavior."
        ),
    )
    parser.add_argument(
        "--include-image-features",
        action="store_true",
        help=(
            "Read stacked crop TIFFs, add mask/image evidence features, and "
            "include them in candidate/pairwise rankers."
        ),
    )
    parser.add_argument(
        "--include-rect-image-features",
        action="store_true",
        help=(
            "Read stacked crop TIFFs, add object-local rectangular mask/image "
            "features, and include them in candidate/pairwise rankers."
        ),
    )
    parser.add_argument(
        "--include-multiscale-rect-image-features",
        action="store_true",
        help=(
            "Read stacked crop TIFFs, add tight/far rectangular mask/image "
            "features, and include them in candidate/pairwise rankers."
        ),
    )
    parser.add_argument(
        "--include-full-image-features",
        action="store_true",
        help=(
            "Read original images, add object-local mask/image features in true "
            "image coordinates, and include them in candidate/pairwise rankers."
        ),
    )
    parser.add_argument(
        "--include-full-image-boundary-noise-features",
        action="store_true",
        help=(
            "Add full-image boundary/noise diagnostics and include them in "
            "candidate/pairwise rankers. This also enables base full-image features."
        ),
    )
    parser.add_argument(
        "--include-crop-geometry-features",
        action="store_true",
        help=(
            "Read original image shapes, add crop padding/image-border geometry "
            "features, and include them in candidate/pairwise rankers."
        ),
    )
    parser.add_argument("--pairwise-min-target-gap", type=float, default=0.03)
    parser.add_argument(
        "--pairwise-logistic-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the pairwise logistic ranker.",
    )
    parser.add_argument(
        "--pairwise-source-identity-blend",
        type=float,
        default=1.0,
        help=(
            "Blend weight for source-identity pairwise probabilities. Values "
            "below 1.0 mix in a source-agnostic model for source-shift robustness."
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

    presets = args.preset or sorted(PRESETS)
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
    if args.use_pairwise_cluster_ranker and args.use_source_identity:
        method_prefix = "soft_hybrid_pairwise_source_ranked"
    elif args.use_pairwise_cluster_ranker:
        method_prefix = "soft_hybrid_pairwise_ranked"
    elif args.use_candidate_ranker and args.use_source_identity:
        method_prefix = "soft_hybrid_source_ranked"
    elif args.use_candidate_ranker:
        method_prefix = "soft_hybrid_ranked"
    else:
        method_prefix = "soft_hybrid"
    summary_records: list[dict[str, object]] = []
    for input_path in args.input:
        features = build_candidate_feature_table(
            [input_path],
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
        feature_path = feature_output_path(input_path, args.output_root)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(feature_path, index=False)
        if args.use_candidate_ranker:
            ranker_model_path = feature_path.with_name(
                f"{feature_path.stem}_candidate_ranker.joblib"
            )
            ranked_feature_path = feature_path.with_name(
                f"{feature_path.stem}_ranked.parquet"
            )
            ranker_payload = train_candidate_ranker(
                features,
                output_model=ranker_model_path,
                config=CandidateRankerConfig(
                    feature_columns=ranker_feature_columns,
                    train_splits=tuple(args.ranker_train_split or ("train",)),
                    calibration_splits=tuple(
                        args.ranker_calibration_split or ("validation",)
                    ),
                    model_type=args.candidate_ranker_model_type,
                    mlp_hidden_layer_sizes=tuple(
                        args.candidate_ranker_mlp_hidden_layer_size or (64, 32)
                    ),
                    mlp_alpha=args.candidate_ranker_mlp_alpha,
                    mlp_max_iter=args.candidate_ranker_mlp_max_iter,
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
            features.to_parquet(ranked_feature_path, index=False)
        if args.use_pairwise_cluster_ranker:
            pairwise_model_path = feature_path.with_name(
                f"{feature_path.stem}_pairwise_cluster_ranker.joblib"
            )
            pairwise_feature_path = feature_path.with_name(
                f"{feature_path.stem}_pairwise_ranked.parquet"
            )
            pairwise_payload = train_cluster_pairwise_ranker(
                features,
                output_model=pairwise_model_path,
                config=ClusterPairwiseRankerConfig(
                    feature_columns=ranker_feature_columns,
                    train_splits=tuple(args.ranker_train_split or ("train",)),
                    use_source_identity=args.use_source_identity,
                    min_target_gap=args.pairwise_min_target_gap,
                    logistic_c=args.pairwise_logistic_c,
                    source_identity_blend=args.pairwise_source_identity_blend,
                ),
            )
            features = score_with_cluster_pairwise_ranker(
                features, model_payload=pairwise_payload
            )
            if args.use_local_image_support_score:
                features = add_local_image_support_score(
                    features, mode=args.local_image_support_mode
                )
            features.to_parquet(pairwise_feature_path, index=False)
        elif args.use_local_image_support_score:
            features = add_local_image_support_score(
                features, mode=args.local_image_support_mode
            )

        for preset in presets:
            weights = PRESETS[preset]
            output_dir = output_subdir(
                input_path, args.output_root, preset, method_prefix
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            selected = run_soft_hybrid_selection(
                features,
                weights=weights,
                max_bbox_iou=args.max_bbox_iou,
                min_center_distance=args.min_center_distance,
                min_score=args.min_score,
            )
            selected_path = output_dir / "selected_candidates.parquet"
            selected.to_parquet(selected_path, index=False)
            eval_df = reconstruct_selected_soft_hybrid_candidates(
                candidate_parquet=input_path,
                selected_candidates_path=selected_path,
                output_dir=output_dir,
            )
            summary_records.extend(
                summarize_eval(
                    eval_df,
                    input_path=input_path,
                    output_dir=output_dir,
                    preset=preset,
                    weights=weights,
                    use_candidate_ranker=args.use_candidate_ranker,
                    use_source_identity=args.use_source_identity,
                )
            )
            print(
                f"{input_path} / {preset}: {len(selected)} selected, "
                f"{len(eval_df)} eval rows"
            )

    summary = pd.DataFrame.from_records(summary_records)
    summary_path = args.output_root / "soft_hybrid_grid_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
