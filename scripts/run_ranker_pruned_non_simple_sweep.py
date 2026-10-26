#!/usr/bin/env python3
"""Run non-SIMPLE backends on existing ranker-pruned candidate pools."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from silver_truth.fusion.crops_experiment import run_crops_fusion_experiment


DEFAULT_MODELS = (
    "MAJORITY_FLAT",
    "THRESHOLD_FLAT",
    "BIC_FLAT_VOTING",
    "THRESHOLD_USER",
    "BIC_WEIGHTED_VOTING",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-root",
        type=Path,
        default=Path("hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe"),
        help="Root containing the existing SIMPLE ranker-pruned fusion runs.",
    )
    parser.add_argument(
        "--best-by-dataset",
        type=Path,
        default=Path(
            "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
            "all_available_simple_policy_best_by_dataset.csv"
        ),
        help="Dataset-level best-policy table from the SIMPLE sweep.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
            "non_simple_best_policy_backend_sweep"
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=DEFAULT_MODELS,
        help="Backend to run. Can be repeated. Defaults to the non-SIMPLE set.",
    )
    parser.add_argument(
        "--weights-column",
        default="pairwise_cluster_score",
        help="Candidate score column for weighted backends.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--mlflow-experiment", default="ranker-pruned-non-simple-sweep")
    parser.add_argument(
        "--mlflow-tracking-path",
        default="data/fusion_experiments/mlruns",
    )
    parser.add_argument("--keep-job-dir", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run outputs that already have comparison rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered jobs without running fusion.",
    )
    args = parser.parse_args()

    models = tuple(args.model or DEFAULT_MODELS)
    jobs = discover_jobs(args.phase1_root, args.best_by_dataset)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit("No existing best-policy ranker-pruned fusion jobs found.")
    if args.dry_run:
        for job in jobs:
            print(
                f"{job['dataset']} {job['fold']} {job['policy']}: "
                f"{job['filtered_parquet']}"
            )
        return

    comparison_paths: list[Path] = []
    for job in jobs:
        output_dir = (
            args.output_root
            / job["dataset"]
            / job["crop_tag"]
            / job["variant"]
            / job["fold"]
            / f"{job['policy']}_non_simple"
        )
        comparison_path = output_dir / "ranker_pruned_fusion_comparison.csv"
        if comparison_path.exists() and not args.overwrite:
            comparison_paths.append(comparison_path)
            print(f"Skipping existing: {comparison_path}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        fusion_dir = output_dir / "fusion"
        print(
            "Running "
            f"{job['dataset']} {job['fold']} {job['policy']} with "
            f"{','.join(models)}"
        )
        result = run_crops_fusion_experiment(
            qa_parquet=job["filtered_parquet"],
            output_dir=fusion_dir,
            models=models,
            weights_column=args.weights_column,
            num_threads=args.num_threads,
            chunk_size=args.chunk_size,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_tracking_path=args.mlflow_tracking_path,
            mlflow_run_name=(
                f"ranker_pruned_non_simple_{job['dataset']}_"
                f"{job['fold']}_{job['policy']}"
            ),
            keep_job_dir=args.keep_job_dir,
            debug=args.debug,
        )
        comparison = build_comparison_rows(
            leaderboard_path=Path(result["leaderboard_path"]),
            previous_comparison_path=job["previous_comparison"],
            pruning_summary_path=job["pruning_summary"],
            policy=job["policy"],
        )
        comparison.to_csv(comparison_path, index=False)
        comparison_paths.append(comparison_path)
        print(f"Wrote comparison rows: {comparison_path}")

    if comparison_paths:
        combined = pd.concat(
            [pd.read_csv(path) for path in comparison_paths],
            ignore_index=True,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        combined_path = args.output_root / "non_simple_best_policy_comparison_rows.csv"
        summary_path = args.output_root / "non_simple_best_policy_dataset_summary.csv"
        best_path = args.output_root / "non_simple_best_policy_best_by_dataset.csv"
        overall_path = args.output_root / "non_simple_best_policy_overall_summary.csv"
        dataset_summary = summarize_dataset(combined)
        best_by_dataset = select_best_backend_by_dataset(dataset_summary)
        overall = summarize_overall(dataset_summary, best_by_dataset)
        combined.to_csv(combined_path, index=False)
        dataset_summary.to_csv(summary_path, index=False)
        best_by_dataset.to_csv(best_path, index=False)
        overall.to_csv(overall_path, index=False)
        print(f"Wrote combined rows: {combined_path}")
        print(f"Wrote dataset summary: {summary_path}")
        print(f"Wrote best-by-dataset summary: {best_path}")
        print(f"Wrote overall summary: {overall_path}")


def discover_jobs(phase1_root: Path, best_by_dataset_path: Path) -> list[dict[str, Any]]:
    best = pd.read_csv(best_by_dataset_path)
    jobs: list[dict[str, Any]] = []
    for row in best.to_dict("records"):
        dataset = str(row["dataset"])
        crop_tag = str(row["crop_tag"])
        variant = str(row["variant"])
        policy = str(row["policy"])
        dataset_root = phase1_root / dataset / crop_tag / variant
        for fold_dir in sorted(dataset_root.glob("fold-*")):
            simple_dir = _find_simple_policy_dir(fold_dir, policy)
            if simple_dir is None:
                continue
            filtered = _find_filtered_parquet(simple_dir, policy)
            previous_comparison = simple_dir / "ranker_pruned_fusion_comparison.csv"
            pruning_summary = simple_dir / "ranker_pruning_summary.csv"
            if not (
                filtered is not None
                and previous_comparison.exists()
                and pruning_summary.exists()
            ):
                continue
            jobs.append(
                {
                    "dataset": dataset,
                    "crop_tag": crop_tag,
                    "variant": variant,
                    "fold": fold_dir.name,
                    "policy": policy,
                    "filtered_parquet": filtered,
                    "previous_comparison": previous_comparison,
                    "pruning_summary": pruning_summary,
                }
            )
    return jobs


def _find_simple_policy_dir(fold_dir: Path, policy: str) -> Path | None:
    matches = sorted(fold_dir.glob(f"{policy}*_simple_escalated"))
    if not matches:
        return None
    return matches[0]


def _find_filtered_parquet(simple_dir: Path, policy: str) -> Path | None:
    matches = sorted(simple_dir.glob(f"*_paper_ready_{policy}.parquet"))
    if matches:
        return matches[0]
    matches = sorted(simple_dir.glob("*.parquet"))
    return matches[0] if matches else None


def build_comparison_rows(
    *,
    leaderboard_path: Path,
    previous_comparison_path: Path,
    pruning_summary_path: Path,
    policy: str,
) -> pd.DataFrame:
    leaderboard = pd.read_csv(leaderboard_path)
    previous = pd.read_csv(previous_comparison_path)
    metadata = previous.iloc[0].to_dict()
    pruning = pd.read_csv(pruning_summary_path)
    stats = pruning.iloc[0].to_dict() if not pruning.empty else {}

    records: list[dict[str, Any]] = []
    for row in leaderboard.to_dict("records"):
        model = str(row.get("model", "unknown"))
        model_key = _method_token(model)
        records.append(
            {
                "dataset": metadata.get("dataset", ""),
                "crop_tag": metadata.get("crop_tag", ""),
                "variant": metadata.get("variant", ""),
                "fold": metadata.get("fold", ""),
                "input_path": metadata.get("input_path", ""),
                "row_family": "full_pipeline",
                "method_key": f"full_pipeline__qa_pruned_{model_key}__{policy}",
                "method_label": f"QA-pruned {model}",
                "policy": policy,
                "fusion_backend": model,
                "mean_iou": _first_finite(
                    row,
                    [
                        "eval_test_mean_jaccard",
                        "selection_mean_jaccard",
                        "eval_overall_mean_jaccard",
                        "mean_jaccard",
                    ],
                ),
                "mean_f1": _first_finite(
                    row,
                    [
                        "eval_test_mean_f1",
                        "selection_mean_f1",
                        "eval_overall_mean_f1",
                        "mean_f1",
                    ],
                ),
                "selection_split": row.get("selection_split"),
                "fusion_success": row.get("fusion_success"),
                "filtered_candidate_rows": stats.get("filtered_candidate_rows"),
                "input_candidate_rows": stats.get("input_candidate_rows"),
                "ranked_candidate_rows": stats.get("ranked_candidate_rows"),
                "matched_ranked_rows": stats.get("matched_ranked_rows"),
                "clusters_before": stats.get("clusters_before"),
                "clusters_after": stats.get("clusters_after"),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_dataset(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "crop_tag", "variant", "policy", "fusion_backend"]
    return (
        rows.groupby(keys, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            mean_filtered_rows=("filtered_candidate_rows", "mean"),
            mean_input_rows=("input_candidate_rows", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "mean_iou"], ascending=[True, False])
    )


def select_best_backend_by_dataset(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    if dataset_summary.empty:
        return dataset_summary.copy()
    return (
        dataset_summary.sort_values(
            ["dataset", "mean_iou", "mean_f1", "fusion_backend"],
            ascending=[True, False, False, True],
        )
        .groupby("dataset", as_index=False, sort=False)
        .first()
        .reset_index(drop=True)
    )


def summarize_overall(
    dataset_summary: pd.DataFrame,
    best_by_dataset: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if not dataset_summary.empty:
        for backend, group in dataset_summary.groupby("fusion_backend", dropna=False):
            records.append(
                {
                    "selection_scope": "fixed_backend",
                    "fusion_backend": backend,
                    "datasets": int(group["dataset"].nunique()),
                    "mean_iou": float(group["mean_iou"].mean()),
                    "mean_f1": float(group["mean_f1"].mean()),
                }
            )
    if not best_by_dataset.empty:
        records.append(
            {
                "selection_scope": "best_backend_by_dataset",
                "fusion_backend": "dataset_selected",
                "datasets": int(best_by_dataset["dataset"].nunique()),
                "mean_iou": float(best_by_dataset["mean_iou"].mean()),
                "mean_f1": float(best_by_dataset["mean_f1"].mean()),
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["selection_scope", "mean_iou"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _method_token(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def _first_finite(record: dict[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            return float(value)
    return float("nan")


if __name__ == "__main__":
    main()
