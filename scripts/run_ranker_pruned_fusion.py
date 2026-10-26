#!/usr/bin/env python3
"""Run crop fusion after QA/ranker-guided candidate pruning."""

from __future__ import annotations

import argparse
from pathlib import Path

from silver_truth.evaluation.ranker_pruned_fusion import (
    PRUNING_POLICIES,
    RankerPruningConfig,
    run_ranker_pruned_fusion,
    write_ranker_pruned_candidate_parquet,
)
from silver_truth.fusion.crops_experiment import ALL_MODELS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a paper-ready candidate parquet using a ranked/selected "
            "candidate table, then run crop-level fusion on the pruned pool."
        )
    )
    parser.add_argument("--candidate-parquet", required=True, type=Path)
    parser.add_argument(
        "--ranked-candidates",
        required=True,
        type=Path,
        help="Ranked feature table or selected_candidates parquet/CSV.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--policy",
        choices=PRUNING_POLICIES,
        default="top2_per_cluster",
        help="Candidate pruning policy.",
    )
    parser.add_argument("--score-column", default=None)
    parser.add_argument("--cluster-column", default=None)
    parser.add_argument("--source-column", default=None)
    parser.add_argument("--support-column", default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--source-top-k", type=int, default=1)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--fusion-safe-margin", type=float, default=0.15)
    parser.add_argument("--fusion-safe-support-quantile", type=float, default=0.50)
    parser.add_argument(
        "--preserve-source-competitors",
        action="store_true",
        help=(
            "Keep original competitor labels in the filtered parquet. By default "
            "kept candidates are reindexed as dense rank slots for fusion."
        ),
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Write the pruned parquet and pruning summary without running fusion.",
    )
    parser.add_argument(
        "--models",
        action="append",
        choices=ALL_MODELS,
        help="Fusion model to run. Can be repeated.",
    )
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--flat-models-only", action="store_true")
    parser.add_argument("--weights-column", default=None)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument(
        "--mlflow-experiment",
        default="ranker-pruned-fusion-crops",
    )
    parser.add_argument(
        "--mlflow-tracking-path",
        default="data/fusion_experiments/mlruns",
    )
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--keep-job-dir", action="store_true")
    parser.add_argument("--job-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = RankerPruningConfig(
        policy=args.policy,
        score_column=args.score_column,
        cluster_column=args.cluster_column,
        source_column=args.source_column,
        support_column=args.support_column,
        top_k=args.top_k,
        source_top_k=args.source_top_k,
        margin=args.margin,
        fusion_safe_margin=args.fusion_safe_margin,
        fusion_safe_support_quantile=args.fusion_safe_support_quantile,
        reindex_competitors_as_rank_slots=not args.preserve_source_competitors,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_parquet = (
        args.output_dir / f"{args.candidate_parquet.stem}_{args.policy}.parquet"
    )
    pruning_summary_path = args.output_dir / "ranker_pruning_summary.csv"

    if args.filter_only:
        _, stats = write_ranker_pruned_candidate_parquet(
            candidate_parquet=args.candidate_parquet,
            ranked_candidates=args.ranked_candidates,
            output_parquet=filtered_parquet,
            config=config,
            summary_path=pruning_summary_path,
        )
        print(f"Wrote filtered parquet: {filtered_parquet}")
        print(f"Wrote pruning summary: {pruning_summary_path}")
        print(stats.to_string(index=False))
        return

    result = run_ranker_pruned_fusion(
        candidate_parquet=args.candidate_parquet,
        ranked_candidates=args.ranked_candidates,
        output_dir=args.output_dir,
        config=config,
        models=tuple(args.models or ()),
        all_models=args.all_models,
        flat_models_only=args.flat_models_only,
        weights_column=args.weights_column,
        num_threads=args.num_threads,
        chunk_size=args.chunk_size,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_tracking_path=args.mlflow_tracking_path,
        skip_fusion=args.skip_fusion,
        keep_job_dir=args.keep_job_dir,
        job_dir=args.job_dir,
        debug=args.debug,
    )

    print("Ranker-pruned fusion finished.")
    print(f"Filtered parquet: {result['filtered_parquet']}")
    print(f"Pruning summary: {result['pruning_summary_path']}")
    print(f"Fusion leaderboard: {result['leaderboard_path']}")
    print(f"Comparison rows: {result['comparison_path']}")


if __name__ == "__main__":
    main()
