#!/usr/bin/env python3
"""Run an internal QA-weighted object-vote ensemble on pruned candidate pools."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from silver_truth.ensemble.qa_weighted_vote import (
    infer_paper_input_metadata_from_pruned_path,
    run_qa_weighted_object_vote,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-parquet",
        action="append",
        type=Path,
        help="Pruned fold candidate parquet. Can be repeated.",
    )
    parser.add_argument(
        "--feature-parquet",
        action="append",
        type=Path,
        help=(
            "Optional feature parquet to merge by cell_id/candidate_id. Can be "
            "repeated once per candidate parquet."
        ),
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        help=(
            "Optional root containing feature parquets at "
            "dataset/crop_tag/variant/fold-*_paper_ready_pairwise_ranked.parquet."
        ),
    )
    parser.add_argument(
        "--discover-best-policy",
        action="store_true",
        help="Discover best-policy pruned pools from the ranker-pruned SIMPLE sweep.",
    )
    parser.add_argument(
        "--phase1-root",
        type=Path,
        default=Path("hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe"),
    )
    parser.add_argument(
        "--best-by-dataset",
        type=Path,
        default=Path(
            "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
            "all_available_simple_policy_best_by_dataset.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for a single input.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("hpc_results/soft_hybrid_probe/qa_weighted_object_vote_probe"),
        help="Root for multi-input or discovered runs.",
    )
    parser.add_argument("--weights-column", default=None)
    parser.add_argument(
        "--boundary-support-column",
        default=None,
        help="Optional per-candidate support column for downweighting mask boundaries.",
    )
    parser.add_argument("--boundary-floor", type=float, default=0.25)
    parser.add_argument(
        "--boundary-floor-column",
        default=None,
        help=(
            "Optional per-candidate floor column. When set, values override "
            "--boundary-floor for each candidate after feature merge."
        ),
    )
    parser.add_argument("--boundary-iterations", type=int, default=1)
    parser.add_argument(
        "--top-k-per-group",
        type=int,
        default=None,
        help=(
            "Keep only the top-k candidates per ranker_cluster_id when present, "
            "otherwise per object-vote group, before weighted voting."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-weight", type=float, default=1e-6)
    parser.add_argument(
        "--method-key",
        default="ensemble_qa_weighted_object_vote",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs without running.",
    )
    args = parser.parse_args()

    jobs = _explicit_jobs(
        args.candidate_parquet or [],
        args.feature_parquet or [],
        feature_root=args.feature_root,
    )
    if args.discover_best_policy:
        jobs.extend(
            _discovered_jobs(
                args.phase1_root,
                args.best_by_dataset,
                feature_root=args.feature_root,
            )
        )
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit("No candidate parquet inputs were provided or discovered.")

    single_input = len(jobs) == 1
    comparison_paths: list[Path] = []
    for job in jobs:
        output_dir = _job_output_dir(
            job,
            output_root=args.output_root,
            output_dir=args.output_dir if single_input else None,
        )
        comparison_path = output_dir / "weighted_object_vote_comparison.csv"
        if args.dry_run:
            print(f"{job['candidate_parquet']} -> {output_dir}")
            continue
        if comparison_path.exists() and not args.overwrite:
            comparison_paths.append(comparison_path)
            print(f"Skipping existing: {comparison_path}")
            continue

        print(f"Running QA-weighted object vote: {job['candidate_parquet']}")
        result = run_qa_weighted_object_vote(
            candidate_parquet=job["candidate_parquet"],
            output_dir=output_dir,
            feature_parquet=job.get("feature_parquet"),
            weights_column=args.weights_column,
            boundary_support_column=args.boundary_support_column,
            boundary_floor=args.boundary_floor,
            boundary_floor_column=args.boundary_floor_column,
            boundary_iterations=args.boundary_iterations,
            top_k_per_group=args.top_k_per_group,
            threshold=args.threshold,
            min_weight=args.min_weight,
            method_key=args.method_key,
        )
        comparison_paths.append(Path(result["comparison_path"]))
        print(f"Wrote comparison row: {result['comparison_path']}")

    if comparison_paths and not args.dry_run:
        _write_combined_outputs(comparison_paths, args.output_root)


def _explicit_jobs(
    paths: list[Path],
    feature_paths: list[Path],
    *,
    feature_root: Path | None = None,
) -> list[dict[str, Any]]:
    if feature_paths and len(feature_paths) not in {1, len(paths)}:
        raise SystemExit(
            "--feature-parquet must be supplied once or once per --candidate-parquet."
        )
    jobs = []
    for index, path in enumerate(paths):
        feature_path = None
        if len(feature_paths) == 1:
            feature_path = feature_paths[0]
        elif feature_paths:
            feature_path = feature_paths[index]
        elif feature_root is not None:
            feature_path = _feature_path_for_candidate(path, feature_root)
        jobs.append(
            {
                "candidate_parquet": path,
                "feature_parquet": feature_path,
                "policy": "",
            }
        )
    return jobs


def _feature_path_for_candidate(candidate: Path, feature_root: Path) -> Path:
    metadata = infer_paper_input_metadata_from_pruned_path(candidate)
    feature_path = (
        feature_root
        / metadata.get("dataset", "")
        / metadata.get("crop_tag", "")
        / metadata.get("variant", "")
        / f"{metadata.get('fold', candidate.stem)}_paper_ready_pairwise_ranked.parquet"
    )
    if not feature_path.exists():
        raise SystemExit(f"Feature parquet not found for {candidate}: {feature_path}")
    return feature_path


def _discovered_jobs(
    phase1_root: Path,
    best_by_dataset: Path,
    *,
    feature_root: Path | None = None,
) -> list[dict[str, Any]]:
    from scripts.run_ranker_pruned_non_simple_sweep import discover_jobs

    jobs = []
    for job in discover_jobs(phase1_root, best_by_dataset):
        candidate = job["filtered_parquet"]
        discovered = {"candidate_parquet": candidate, "policy": job["policy"]}
        if feature_root is not None:
            discovered["feature_parquet"] = _feature_path_for_candidate(
                candidate, feature_root
            )
        jobs.append(discovered)
    return jobs


def _job_output_dir(
    job: dict[str, Any],
    *,
    output_root: Path,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir

    candidate = Path(job["candidate_parquet"])
    metadata = infer_paper_input_metadata_from_pruned_path(candidate)
    dataset = metadata.get("dataset") or "unknown_dataset"
    crop_tag = metadata.get("crop_tag") or "unknown_crop"
    variant = metadata.get("variant") or "unknown_variant"
    fold = metadata.get("fold") or candidate.stem
    policy = str(job.get("policy") or _policy_from_candidate_name(candidate))
    run_name = "weighted_object_vote" if not policy else f"{policy}_weighted_object_vote"
    return output_root / dataset / crop_tag / variant / fold / run_name


def _policy_from_candidate_name(path: Path) -> str:
    stem = path.stem
    marker = "_paper_ready_"
    if marker in stem:
        return stem.rsplit(marker, 1)[-1]
    return ""


def _write_combined_outputs(comparison_paths: list[Path], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [pd.read_csv(path) for path in comparison_paths],
        ignore_index=True,
    )
    combined_path = output_root / "qa_weighted_object_vote_comparison_rows.csv"
    summary_path = output_root / "qa_weighted_object_vote_dataset_summary.csv"
    overall_path = output_root / "qa_weighted_object_vote_overall_summary.csv"
    summary = _summarize_dataset(combined)
    overall = _summarize_overall(summary)
    combined.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)
    overall.to_csv(overall_path, index=False)
    print(f"Wrote combined rows: {combined_path}")
    print(f"Wrote dataset summary: {summary_path}")
    print(f"Wrote overall summary: {overall_path}")


def _summarize_dataset(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "crop_tag",
        "variant",
        "row_family",
        "method_key",
        "method_label",
        "weights_column",
        "boundary_support_column",
        "boundary_floor",
        "boundary_floor_column",
        "boundary_iterations",
        "top_k_per_group",
        "threshold",
    ]
    present = [column for column in group_cols if column in rows.columns]
    return (
        rows.groupby(present, dropna=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            n_folds=("fold", "nunique"),
            n_cells=("n_cells", "sum"),
            mean_candidates_used=("mean_candidates_used", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "mean_iou"], ascending=[True, False])
    )


def _summarize_overall(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "row_family",
        "method_key",
        "method_label",
        "weights_column",
        "boundary_support_column",
        "boundary_floor",
        "boundary_floor_column",
        "boundary_iterations",
        "top_k_per_group",
        "threshold",
    ]
    present = [column for column in group_cols if column in summary.columns]
    return (
        summary.groupby(present, dropna=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            n_datasets=("dataset", "nunique"),
            n_folds=("n_folds", "sum"),
            n_cells=("n_cells", "sum"),
            mean_candidates_used=("mean_candidates_used", "mean"),
        )
        .reset_index()
        .sort_values("mean_iou", ascending=False)
    )


if __name__ == "__main__":
    main()
