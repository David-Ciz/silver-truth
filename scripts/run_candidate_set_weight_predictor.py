#!/usr/bin/env python3
"""Train and apply a candidate-set weight predictor to feature parquets."""

from __future__ import annotations

import argparse
from pathlib import Path

from silver_truth.evaluation.candidate_set_weight import (
    CandidateSetWeightConfig,
    write_leave_one_dataset_out_weighted_features,
    write_leave_target_video_out_weighted_features,
    write_candidate_set_weighted_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-parquet", action="append", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument(
        "--feature-pattern",
        default="**/*_paper_ready_pairwise_ranked.parquet",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument(
        "--leave-one-dataset-out",
        action="store_true",
        help="Score each dataset with a model trained on the other datasets.",
    )
    parser.add_argument(
        "--leave-target-video-out",
        action="store_true",
        help=(
            "Score each dataset/fold with a model that excludes all rows from "
            "that dataset's held-out test sequence."
        ),
    )
    parser.add_argument("--train-split", action="append", default=None)
    parser.add_argument(
        "--model-type",
        choices=("mlp", "ridge", "set_mlp", "deep_sets"),
        default="mlp",
    )
    parser.add_argument(
        "--categorical-column",
        action="append",
        default=None,
        help=(
            "Categorical feature column for the predictor. Can be repeated. "
            "Defaults to dataset and candidate_source."
        ),
    )
    parser.add_argument(
        "--mlp-hidden-layer-size",
        action="append",
        type=int,
        help="Hidden layer size for MLP candidate-set predictor. Can be repeated.",
    )
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--target-temperature", type=float, default=0.08)
    parser.add_argument("--min-score", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--deep-sets-hidden-dim", type=int, default=64)
    parser.add_argument("--deep-sets-epochs", type=int, default=120)
    parser.add_argument("--deep-sets-learning-rate", type=float, default=1e-3)
    parser.add_argument("--deep-sets-weight-decay", type=float, default=1e-4)
    parser.add_argument("--deep-sets-batch-clusters", type=int, default=32)
    parser.add_argument("--deep-sets-device", default="auto")
    parser.add_argument("--deep-sets-bad-mass-penalty-weight", type=float, default=0.0)
    parser.add_argument("--deep-sets-bad-mass-margin", type=float, default=0.10)
    parser.add_argument(
        "--deep-sets-oversize-bad-mass-penalty-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--deep-sets-oversize-bad-mass-margin", type=float, default=0.10)
    parser.add_argument(
        "--deep-sets-oversize-area-column",
        default="full_mask_area_log_ratio_to_cluster_median",
    )
    parser.add_argument(
        "--deep-sets-oversize-area-log-ratio-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--categorical-dropout-column",
        action="append",
        default=None,
        help=(
            "Categorical column to randomly mask during training. Can be repeated. "
            "Use with --categorical-dropout-probability."
        ),
    )
    parser.add_argument(
        "--categorical-dropout-probability",
        type=float,
        default=0.0,
        help="Probability of masking configured categorical dropout columns.",
    )
    parser.add_argument(
        "--dataset-dropout-prob",
        type=float,
        default=None,
        help=(
            "Convenience alias for masking the dataset categorical input during "
            "training. Adds dataset to categorical dropout columns when needed."
        ),
    )
    args = parser.parse_args()

    inputs = list(args.feature_parquet or [])
    relative_root = args.feature_root if args.feature_root is not None else None
    if args.feature_root is not None:
        inputs.extend(sorted(args.feature_root.glob(args.feature_pattern)))
    if not inputs:
        raise SystemExit("No feature parquets were provided or discovered.")

    config = _config_from_args(args)
    if args.leave_one_dataset_out and args.leave_target_video_out:
        raise SystemExit(
            "Choose only one of --leave-one-dataset-out or --leave-target-video-out."
        )
    if args.leave_target_video_out:
        model_root = args.output_model or (args.output_root / "models")
        summary = write_leave_target_video_out_weighted_features(
            inputs,
            output_root=args.output_root,
            relative_root=relative_root,
            output_model_root=model_root,
            config=config,
        )
    elif args.leave_one_dataset_out:
        model_root = args.output_model or (args.output_root / "models")
        summary = write_leave_one_dataset_out_weighted_features(
            inputs,
            output_root=args.output_root,
            relative_root=relative_root,
            output_model_root=model_root,
            config=config,
        )
    else:
        summary = write_candidate_set_weighted_features(
            inputs,
            output_root=args.output_root,
            relative_root=relative_root,
            output_model=args.output_model,
            config=config,
        )
    summary_path = args.output_root / "candidate_set_weight_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


def _config_from_args(args: argparse.Namespace) -> CandidateSetWeightConfig:
    categorical_dropout_columns = tuple(args.categorical_dropout_column or ())
    categorical_dropout_probability = args.categorical_dropout_probability
    dataset_dropout_prob = getattr(args, "dataset_dropout_prob", None)
    if dataset_dropout_prob is not None:
        categorical_dropout_probability = dataset_dropout_prob
        if "dataset" not in categorical_dropout_columns:
            categorical_dropout_columns = (*categorical_dropout_columns, "dataset")

    return CandidateSetWeightConfig(
        train_splits=tuple(args.train_split or ("train", "validation")),
        model_type=args.model_type,
        categorical_columns=tuple(
            args.categorical_column or ("dataset", "candidate_source")
        ),
        mlp_hidden_layer_sizes=tuple(args.mlp_hidden_layer_size or (64, 32)),
        mlp_alpha=args.mlp_alpha,
        mlp_max_iter=args.mlp_max_iter,
        target_temperature=args.target_temperature,
        min_score=args.min_score,
        random_state=args.random_state,
        categorical_dropout_columns=categorical_dropout_columns,
        categorical_dropout_probability=categorical_dropout_probability,
        deep_sets_hidden_dim=getattr(args, "deep_sets_hidden_dim", 64),
        deep_sets_epochs=getattr(args, "deep_sets_epochs", 120),
        deep_sets_learning_rate=getattr(args, "deep_sets_learning_rate", 1e-3),
        deep_sets_weight_decay=getattr(args, "deep_sets_weight_decay", 1e-4),
        deep_sets_batch_clusters=getattr(args, "deep_sets_batch_clusters", 32),
        deep_sets_device=getattr(args, "deep_sets_device", "auto"),
        deep_sets_bad_mass_penalty_weight=getattr(
            args, "deep_sets_bad_mass_penalty_weight", 0.0
        ),
        deep_sets_bad_mass_margin=getattr(args, "deep_sets_bad_mass_margin", 0.10),
        deep_sets_oversize_bad_mass_penalty_weight=getattr(
            args, "deep_sets_oversize_bad_mass_penalty_weight", 0.0
        ),
        deep_sets_oversize_bad_mass_margin=getattr(
            args, "deep_sets_oversize_bad_mass_margin", 0.10
        ),
        deep_sets_oversize_area_column=getattr(
            args,
            "deep_sets_oversize_area_column",
            "full_mask_area_log_ratio_to_cluster_median",
        ),
        deep_sets_oversize_area_log_ratio_threshold=getattr(
            args, "deep_sets_oversize_area_log_ratio_threshold", 0.25
        ),
    )


if __name__ == "__main__":
    main()
