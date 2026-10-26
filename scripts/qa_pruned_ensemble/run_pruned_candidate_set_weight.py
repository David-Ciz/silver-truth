#!/usr/bin/env python3
"""Train learned candidate-set weights on QA-pruned candidate pools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    add_repo_paths,
    combine_comparison_paths,
    normalize_weighted_vote_outputs,
)

add_repo_paths()

from silver_truth.ensemble.qa_weighted_vote import run_qa_weighted_object_vote  # noqa: E402
from silver_truth.evaluation.candidate_set_weight import (  # noqa: E402
    CandidateSetWeightConfig,
    DEFAULT_CANDIDATE_SET_WEIGHT_FEATURE_COLUMNS,
    write_leave_target_video_out_weighted_features,
)


EXPLICIT_QA_FEATURE_COLUMNS = (
    "qa_score",
    "qa_rank_in_image",
    "ranker_cluster_qa_rank",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "soft_gate_sweep" / "boundary_floor075",
    )
    parser.add_argument(
        "--candidate-run",
        default="clipped_score_boundary_floor075_thr050",
        help="Prepared soft-gate run directory to discover under --candidate-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "learned_candidate_set_weight",
    )
    parser.add_argument(
        "--model-type",
        choices=("mlp", "ridge", "set_mlp", "deep_sets"),
        default="mlp",
    )
    parser.add_argument("--train-split", action="append", default=None)
    parser.add_argument("--categorical-column", action="append", default=None)
    parser.add_argument("--mlp-hidden-layer-size", action="append", type=int)
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
        "--exclude-feature-column",
        action="append",
        default=None,
        help="Numeric candidate-set feature column to remove before training.",
    )
    parser.add_argument(
        "--exclude-explicit-qa-features",
        action="store_true",
        help=(
            "Remove explicit QA columns from the candidate-set weight model: "
            "qa_score, qa_rank_in_image, ranker_cluster_qa_rank."
        ),
    )
    parser.add_argument(
        "--run-label-suffix",
        default="",
        help="Optional suffix for method keys and output directories.",
    )
    parser.add_argument("--vote-threshold", action="append", type=float, default=None)
    parser.add_argument(
        "--weight-transform",
        choices=(
            "clipped_score",
            "raw_positive",
            "raw_cluster_minmax",
            "raw_cluster_softmax",
        ),
        default="clipped_score",
        help=(
            "Weight column transform used for object vote after learned scoring. "
            "clipped_score uses candidate_set_weight_score."
        ),
    )
    parser.add_argument("--weight-temperature", type=float, default=1.0)
    parser.add_argument("--boundary-support-column", default="local_image_support_score")
    parser.add_argument("--boundary-floor", type=float, default=0.75)
    parser.add_argument("--boundary-floor-column", default=None)
    parser.add_argument("--boundary-iterations", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = discover_prepared_candidate_jobs(args.candidate_root, args.candidate_run)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit("No prepared QA-pruned candidate parquets were discovered.")

    feature_paths = [Path(job["prepared_parquet"]) for job in jobs]
    scored_root = args.output_root / _run_name(args)
    model_root = scored_root / "models"
    summary_path = scored_root / "candidate_set_weight_summary.csv"
    if args.dry_run:
        for job in jobs:
            print(f"{job['dataset']} {job['fold']}: {job['prepared_parquet']}")
        return

    if summary_path.exists() and not args.overwrite:
        summary = pd.read_csv(summary_path)
        print(f"Using existing learned-weight summary: {summary_path}")
    else:
        summary = write_leave_target_video_out_weighted_features(
            feature_paths,
            output_root=scored_root / "scored_features",
            relative_root=args.candidate_root,
            output_model_root=model_root,
            config=_config_from_args(args),
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Wrote learned-weight summary: {summary_path}")

    output_by_input = {
        str(row["feature_path"]): Path(row["output_path"])
        for row in summary.to_dict("records")
    }
    vote_thresholds = args.vote_threshold or [0.50]
    comparison_paths: list[Path] = []
    for job in jobs:
        scored_path = output_by_input.get(str(job["prepared_parquet"]))
        if scored_path is None:
            raise SystemExit(f"No scored feature path for {job['prepared_parquet']}")
        vote_path, weights_column = _prepare_vote_feature_path(
            scored_path=Path(scored_path),
            scored_root=scored_root,
            candidate_root=args.candidate_root,
            weight_transform=args.weight_transform,
            weight_temperature=args.weight_temperature,
            min_score=args.min_score,
        )
        for threshold in vote_thresholds:
            output_dir = _vote_output_dir(
                scored_root / "object_vote",
                job=job,
                threshold=threshold,
                weight_transform=args.weight_transform,
                weight_temperature=args.weight_temperature,
            )
            comparison_path = output_dir / "weighted_object_vote_comparison.csv"
            if comparison_path.exists() and not args.overwrite:
                comparison_paths.append(comparison_path)
                print(f"Skipping existing: {comparison_path}")
                continue

            print(
                "Running learned QA-pruned ENSEMBLE object vote: "
                f"{job['dataset']} {job['fold']} {job['candidate_run']} "
                f"thr={threshold:.2f}"
            )
            run_qa_weighted_object_vote(
                candidate_parquet=vote_path,
                output_dir=output_dir,
                weights_column=weights_column,
                boundary_support_column=args.boundary_support_column,
                boundary_floor=args.boundary_floor,
                boundary_floor_column=args.boundary_floor_column,
                boundary_iterations=args.boundary_iterations,
                threshold=threshold,
                method_key=_method_key(args),
            )
            comparison_paths.append(
                normalize_weighted_vote_outputs(
                    output_dir=output_dir,
                    job={**job, "filtered_parquet": vote_path},
                    ensemble_variant=_ensemble_variant(args),
                    method_key=_method_key(args),
                    method_label=_method_label(args),
                    result_family="qa_pruned_ensemble_candidate_set_weight",
                    soft_gate=str(job.get("soft_gate", "")),
                )
            )

    outputs = combine_comparison_paths(
        comparison_paths,
        output_root=scored_root,
        prefix="qa_pruned_ensemble_candidate_set_weight",
    )
    print(f"Wrote learned QA-pruned ENSEMBLE tables to {scored_root}")
    print(outputs["overall_summary"])


def discover_prepared_candidate_jobs(
    candidate_root: Path,
    candidate_run: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in sorted(candidate_root.glob(f"**/{candidate_run}/prepared_soft_gate_candidates.parquet")):
        try:
            rel = path.relative_to(candidate_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 7:
            continue
        jobs.append(
            {
                "dataset": parts[0],
                "crop_tag": parts[1],
                "variant": parts[2],
                "fold": parts[3],
                "policy": parts[4],
                "candidate_run": parts[5],
                "soft_gate": _soft_gate_from_candidate_run(parts[5]),
                "prepared_parquet": path,
            }
        )
    return jobs


def _config_from_args(args: argparse.Namespace) -> CandidateSetWeightConfig:
    excluded = set(args.exclude_feature_column or ())
    if args.exclude_explicit_qa_features:
        excluded.update(EXPLICIT_QA_FEATURE_COLUMNS)
    feature_columns = tuple(
        column
        for column in DEFAULT_CANDIDATE_SET_WEIGHT_FEATURE_COLUMNS
        if column not in excluded
    )
    return CandidateSetWeightConfig(
        train_splits=tuple(args.train_split or ("train",)),
        feature_columns=feature_columns,
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
        deep_sets_hidden_dim=args.deep_sets_hidden_dim,
        deep_sets_epochs=args.deep_sets_epochs,
        deep_sets_learning_rate=args.deep_sets_learning_rate,
        deep_sets_weight_decay=args.deep_sets_weight_decay,
        deep_sets_batch_clusters=args.deep_sets_batch_clusters,
        deep_sets_device=args.deep_sets_device,
        deep_sets_bad_mass_penalty_weight=args.deep_sets_bad_mass_penalty_weight,
        deep_sets_bad_mass_margin=args.deep_sets_bad_mass_margin,
    )


def _run_name(args: argparse.Namespace) -> str:
    suffix = args.run_label_suffix.strip("_")
    base = f"{args.model_type}_ltv_{args.candidate_run}"
    return base if not suffix else f"{base}_{suffix}"


def _ensemble_variant(args: argparse.Namespace) -> str:
    suffix = args.run_label_suffix.strip("_")
    base = (
        f"candidate_set_{args.model_type}_{args.candidate_run}"
        f"_{_weight_transform_token(args)}"
    )
    return base if not suffix else f"{base}_{suffix}"


def _method_key(args: argparse.Namespace) -> str:
    suffix = args.run_label_suffix.strip("_")
    base = (
        f"ensemble_qa_pruned_candidate_set_{args.model_type}_object_vote"
        f"_{_weight_transform_token(args)}"
    )
    return base if not suffix else f"{base}_{suffix}"


def _method_label(args: argparse.Namespace) -> str:
    model_label = {
        "mlp": "MLP",
        "ridge": "ridge",
        "set_mlp": "set-MLP",
        "deep_sets": "Deep Sets",
    }[args.model_type]
    suffix = args.run_label_suffix.strip("_")
    detail = f" ({suffix})" if suffix else ""
    transform = args.weight_transform.replace("_", " ")
    return f"QA-pruned candidate-set {model_label} ENSEMBLE, {transform}{detail}"


def _vote_output_dir(
    output_root: Path,
    *,
    job: dict[str, Any],
    threshold: float,
    weight_transform: str,
    weight_temperature: float,
) -> Path:
    threshold_token = f"thr{int(round(threshold * 100)):03d}"
    transform_token = _weight_transform_token_values(
        weight_transform,
        weight_temperature,
    )
    return (
        output_root
        / str(job["dataset"])
        / str(job["crop_tag"])
        / str(job["variant"])
        / str(job["fold"])
        / str(job["policy"])
        / str(job["candidate_run"])
        / transform_token
        / threshold_token
    )


def _soft_gate_from_candidate_run(candidate_run: str) -> str:
    if candidate_run.startswith("clipped_score"):
        return "clipped_score"
    if candidate_run.startswith("rank_decay"):
        return "rank_decay"
    if candidate_run.startswith("score_rank_product"):
        return "score_rank_product"
    return ""


def _prepare_vote_feature_path(
    *,
    scored_path: Path,
    scored_root: Path,
    candidate_root: Path,
    weight_transform: str,
    weight_temperature: float,
    min_score: float,
) -> tuple[Path, str]:
    if weight_transform == "clipped_score":
        return scored_path, "candidate_set_weight_score"
    column = f"candidate_set_weight_{_weight_transform_token_values(weight_transform, weight_temperature)}"
    output_path = _calibrated_output_path(
        scored_path=scored_path,
        scored_root=scored_root,
        candidate_root=candidate_root,
        weight_transform=weight_transform,
        weight_temperature=weight_temperature,
    )
    if output_path.exists():
        return output_path, column
    frame = pd.read_parquet(scored_path)
    frame[column] = _calibrated_weight_values(
        frame,
        weight_transform=weight_transform,
        weight_temperature=weight_temperature,
        min_score=min_score,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path, column


def _calibrated_output_path(
    *,
    scored_path: Path,
    scored_root: Path,
    candidate_root: Path,
    weight_transform: str,
    weight_temperature: float,
) -> Path:
    scored_features_root = scored_root / "scored_features"
    try:
        rel = scored_path.relative_to(scored_features_root)
    except ValueError:
        rel = scored_path.relative_to(candidate_root)
    return (
        scored_root
        / "calibrated_features"
        / _weight_transform_token_values(weight_transform, weight_temperature)
        / rel
    )


def _calibrated_weight_values(
    frame: pd.DataFrame,
    *,
    weight_transform: str,
    weight_temperature: float,
    min_score: float,
) -> pd.Series:
    if "candidate_set_weight_raw_score" not in frame.columns:
        raise ValueError("Missing candidate_set_weight_raw_score for calibration.")
    raw = pd.to_numeric(frame["candidate_set_weight_raw_score"], errors="coerce")
    if weight_transform == "raw_positive":
        values = raw.clip(lower=min_score)
    elif weight_transform == "raw_cluster_minmax":
        values = _cluster_minmax(raw, frame=frame, min_score=min_score)
    elif weight_transform == "raw_cluster_softmax":
        values = _cluster_softmax(
            raw,
            frame=frame,
            temperature=weight_temperature,
            min_score=min_score,
        )
    else:
        raise ValueError(f"Unknown weight transform: {weight_transform}")
    return pd.Series(values, index=frame.index).fillna(min_score).clip(lower=min_score)


def _cluster_minmax(
    raw: pd.Series,
    *,
    frame: pd.DataFrame,
    min_score: float,
) -> pd.Series:
    groups = _cluster_groups(frame)
    group_min = raw.groupby(groups, dropna=False).transform("min")
    group_max = raw.groupby(groups, dropna=False).transform("max")
    denom = (group_max - group_min).replace(0, np.nan)
    return ((raw - group_min) / denom).fillna(1.0).clip(lower=min_score, upper=1.0)


def _cluster_softmax(
    raw: pd.Series,
    *,
    frame: pd.DataFrame,
    temperature: float,
    min_score: float,
) -> pd.Series:
    if temperature <= 0:
        raise ValueError("weight_temperature must be positive.")
    groups = _cluster_groups(frame)
    centered = raw - raw.groupby(groups, dropna=False).transform("max")
    values = np.exp((centered / temperature).clip(lower=-50.0, upper=0.0))
    return pd.Series(values, index=frame.index).clip(lower=min_score)


def _cluster_groups(frame: pd.DataFrame) -> pd.Series:
    if "ranker_cluster_id" in frame.columns:
        return frame["ranker_cluster_id"].astype(str)
    columns = [
        column
        for column in ("campaign_number", "original_image_key", "label")
        if column in frame.columns
    ]
    if not columns:
        return pd.Series("all_candidates", index=frame.index)
    return frame[columns].astype(str).agg("__".join, axis=1)


def _weight_transform_token(args: argparse.Namespace) -> str:
    return _weight_transform_token_values(
        args.weight_transform,
        args.weight_temperature,
    )


def _weight_transform_token_values(
    weight_transform: str,
    weight_temperature: float,
) -> str:
    if weight_transform != "raw_cluster_softmax":
        return weight_transform
    temp = f"{weight_temperature:.3f}".replace(".", "p").replace("-", "m")
    return f"{weight_transform}_t{temp}"


if __name__ == "__main__":
    main()
