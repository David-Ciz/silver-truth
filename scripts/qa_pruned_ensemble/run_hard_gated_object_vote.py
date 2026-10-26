#!/usr/bin/env python3
"""Run QA-pruned ENSEMBLE hard-gated object-vote baselines."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_BEST_BY_DATASET,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PHASE1_ROOT,
    add_repo_paths,
    combine_comparison_paths,
    discover_best_policy_jobs,
    feature_path_for_job,
    normalize_weighted_vote_outputs,
    output_dir_for_job,
)

add_repo_paths()

from silver_truth.ensemble.qa_weighted_vote import run_qa_weighted_object_vote  # noqa: E402


@dataclass(frozen=True)
class HardGateVariant:
    name: str
    weights_column: str
    method_key: str
    method_label: str
    boundary: bool = False


VARIANTS = {
    "uniform": HardGateVariant(
        name="uniform",
        weights_column="uniform",
        method_key="ensemble_qa_pruned_hard_gate_uniform_object_vote",
        method_label="QA-pruned uniform ENSEMBLE",
    ),
    "qa_weighted": HardGateVariant(
        name="qa_weighted",
        weights_column="pairwise_cluster_score",
        method_key="ensemble_qa_pruned_hard_gate_weighted_object_vote",
        method_label="QA-pruned QA-weighted ENSEMBLE",
    ),
    "boundary_uniform": HardGateVariant(
        name="boundary_uniform",
        weights_column="uniform",
        method_key="ensemble_qa_pruned_hard_gate_boundary_uniform_object_vote",
        method_label="QA-pruned boundary-aware uniform ENSEMBLE",
        boundary=True,
    ),
    "boundary_qa_weighted": HardGateVariant(
        name="boundary_qa_weighted",
        weights_column="pairwise_cluster_score",
        method_key="ensemble_qa_pruned_hard_gate_boundary_weighted_object_vote",
        method_label="QA-pruned boundary-aware QA-weighted ENSEMBLE",
        boundary=True,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--best-by-dataset", type=Path, default=DEFAULT_BEST_BY_DATASET)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "hard_gated_object_vote",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        help="Variant to run. Defaults to uniform and QA-weighted.",
    )
    parser.add_argument("--threshold", action="append", type=float, default=None)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--boundary-support-column", default="local_image_support_score")
    parser.add_argument("--boundary-floor", type=float, default=0.25)
    parser.add_argument("--boundary-floor-column", default=None)
    parser.add_argument("--boundary-iterations", type=int, default=1)
    parser.add_argument("--min-weight", type=float, default=1e-6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variant_names = args.variant or ["uniform", "qa_weighted"]
    thresholds = args.threshold or [0.50]
    jobs = discover_best_policy_jobs(args.phase1_root, args.best_by_dataset)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit("No best-policy QA-pruned candidate pools were discovered.")

    comparison_paths: list[Path] = []
    for job in jobs:
        for variant_name in variant_names:
            variant = VARIANTS[variant_name]
            for threshold in thresholds:
                output_dir = output_dir_for_job(
                    output_root=args.output_root,
                    job=job,
                    ensemble_variant=variant.name,
                    threshold=threshold,
                )
                comparison_path = output_dir / "weighted_object_vote_comparison.csv"
                if args.dry_run:
                    print(
                        f"{job['dataset']} {job['fold']} {job['policy']} "
                        f"{variant.name} thr={threshold:.2f}: {job['filtered_parquet']}"
                    )
                    continue
                if comparison_path.exists() and not args.overwrite:
                    comparison_paths.append(comparison_path)
                    print(f"Skipping existing: {comparison_path}")
                    continue

                feature_path = None
                boundary_column = None
                if variant.boundary:
                    feature_path = feature_path_for_job(job, args.feature_root)
                    if not feature_path.exists():
                        raise SystemExit(f"Feature parquet not found: {feature_path}")
                    boundary_column = args.boundary_support_column

                print(
                    f"Running hard-gated QA-pruned ENSEMBLE: {job['dataset']} "
                    f"{job['fold']} {job['policy']} {variant.name}"
                )
                run_qa_weighted_object_vote(
                    candidate_parquet=Path(job["filtered_parquet"]),
                    output_dir=output_dir,
                    feature_parquet=feature_path,
                    weights_column=variant.weights_column,
                    boundary_support_column=boundary_column,
                    boundary_floor=args.boundary_floor,
                    boundary_floor_column=args.boundary_floor_column,
                    boundary_iterations=args.boundary_iterations,
                    threshold=threshold,
                    min_weight=args.min_weight,
                    method_key=variant.method_key,
                )
                comparison_paths.append(
                    normalize_weighted_vote_outputs(
                        output_dir=output_dir,
                        job=job,
                        ensemble_variant=variant.name,
                        method_key=variant.method_key,
                        method_label=variant.method_label,
                        result_family="qa_pruned_ensemble_hard_gate",
                    )
                )

    if comparison_paths and not args.dry_run:
        outputs = combine_comparison_paths(
            comparison_paths,
            output_root=args.output_root,
            prefix="qa_pruned_ensemble_hard_gate",
            context_path=args.best_by_dataset,
        )
        print(f"Wrote hard-gated QA-pruned ENSEMBLE tables to {args.output_root}")
        print(outputs["overall_summary"])


if __name__ == "__main__":
    main()
