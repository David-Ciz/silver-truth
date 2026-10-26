#!/usr/bin/env python3
"""Run soft QA-passthrough object-vote ENSEMBLE variants."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PHASE1_ROOT,
    add_repo_paths,
    combine_comparison_paths,
    discover_policy_jobs,
    normalize_weighted_vote_outputs,
    output_dir_for_job,
)

add_repo_paths()

from silver_truth.ensemble.qa_weighted_vote import run_qa_weighted_object_vote  # noqa: E402


SOFT_WEIGHT_COLUMN = "_qa_soft_gate_weight"


@dataclass(frozen=True)
class SoftGateSpec:
    name: str
    method_key: str
    method_label: str


SOFT_GATES = {
    "clipped_score": SoftGateSpec(
        name="clipped_score",
        method_key="ensemble_qa_pruned_soft_gate_clipped_score_object_vote",
        method_label="QA-pruned soft-gated ENSEMBLE, clipped score",
    ),
    "sigmoid_score": SoftGateSpec(
        name="sigmoid_score",
        method_key="ensemble_qa_pruned_soft_gate_sigmoid_score_object_vote",
        method_label="QA-pruned soft-gated ENSEMBLE, sigmoid score",
    ),
    "rank_decay": SoftGateSpec(
        name="rank_decay",
        method_key="ensemble_qa_pruned_soft_gate_rank_decay_object_vote",
        method_label="QA-pruned soft-gated ENSEMBLE, rank decay",
    ),
    "score_rank_product": SoftGateSpec(
        name="score_rank_product",
        method_key="ensemble_qa_pruned_soft_gate_score_rank_product_object_vote",
        method_label="QA-pruned soft-gated ENSEMBLE, score-rank product",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "soft_gated_object_vote",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help=(
            "QA-pruned policy to use as the soft candidate pool. Can be repeated. "
            "Defaults to source_balanced_topk."
        ),
    )
    parser.add_argument(
        "--gate",
        action="append",
        choices=sorted(SOFT_GATES),
        default=None,
        help="Soft gate transform. Defaults to clipped_score and rank_decay.",
    )
    parser.add_argument("--score-column", default="pairwise_cluster_score")
    parser.add_argument("--rank-column", default="pairwise_cluster_rank")
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--rank-temperature", type=float, default=2.0)
    parser.add_argument("--min-gate", type=float, default=1e-3)
    parser.add_argument(
        "--boundary-support-column",
        default=None,
        help=(
            "Optional per-candidate support column for boundary-aware soft voting, "
            "for example local_image_support_score."
        ),
    )
    parser.add_argument("--boundary-floor", type=float, default=0.25)
    parser.add_argument("--boundary-floor-column", default=None)
    parser.add_argument("--boundary-iterations", type=int, default=1)
    parser.add_argument(
        "--run-label-suffix",
        default="",
        help=(
            "Optional suffix appended to the ensemble variant so parameter "
            "sweeps remain distinct in summary tables."
        ),
    )
    parser.add_argument("--vote-threshold", action="append", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    policies = args.policy or ["source_balanced_topk"]
    gate_names = args.gate or ["clipped_score", "rank_decay"]
    vote_thresholds = args.vote_threshold or [0.50]
    jobs = discover_policy_jobs(args.phase1_root, policies=policies)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit("No QA-pruned candidate pools were discovered.")

    comparison_paths: list[Path] = []
    for job in jobs:
        for gate_name in gate_names:
            gate = SOFT_GATES[gate_name]
            ensemble_variant = _variant_name(gate.name, args.run_label_suffix)
            method_key = _method_key(gate.method_key, args.run_label_suffix)
            method_label = _method_label(gate.method_label, args.run_label_suffix)
            for vote_threshold in vote_thresholds:
                output_dir = output_dir_for_job(
                    output_root=args.output_root,
                    job=job,
                    ensemble_variant=ensemble_variant,
                    threshold=vote_threshold,
                )
                comparison_path = output_dir / "weighted_object_vote_comparison.csv"
                if args.dry_run:
                    print(
                        f"{job['dataset']} {job['fold']} {job['policy']} "
                        f"{ensemble_variant} vote_thr={vote_threshold:.2f}: "
                        f"{job['filtered_parquet']}"
                    )
                    continue
                if comparison_path.exists() and not args.overwrite:
                    comparison_paths.append(comparison_path)
                    print(f"Skipping existing: {comparison_path}")
                    continue

                print(
                    f"Running soft-gated QA-pruned ENSEMBLE: {job['dataset']} "
                    f"{job['fold']} {job['policy']} {ensemble_variant}"
                )
                prepared = _prepare_soft_gate_parquet(
                    source=Path(job["filtered_parquet"]),
                    output_dir=output_dir,
                    gate=gate.name,
                    score_column=args.score_column,
                    rank_column=args.rank_column,
                    gate_threshold=args.gate_threshold,
                    temperature=args.temperature,
                    rank_temperature=args.rank_temperature,
                    min_gate=args.min_gate,
                )
                run_qa_weighted_object_vote(
                    candidate_parquet=prepared,
                    output_dir=output_dir,
                    weights_column=SOFT_WEIGHT_COLUMN,
                    boundary_support_column=args.boundary_support_column,
                    boundary_floor=args.boundary_floor,
                    boundary_floor_column=args.boundary_floor_column,
                    boundary_iterations=args.boundary_iterations,
                    threshold=vote_threshold,
                    method_key=method_key,
                )
                comparison_paths.append(
                    normalize_weighted_vote_outputs(
                        output_dir=output_dir,
                        job={**job, "filtered_parquet": prepared},
                        ensemble_variant=ensemble_variant,
                        method_key=method_key,
                        method_label=method_label,
                        result_family="qa_pruned_ensemble_soft_gate",
                        soft_gate=gate.name,
                    )
                )

    if comparison_paths and not args.dry_run:
        outputs = combine_comparison_paths(
            comparison_paths,
            output_root=args.output_root,
            prefix="qa_pruned_ensemble_soft_gate",
        )
        print(f"Wrote soft-gated QA-pruned ENSEMBLE tables to {args.output_root}")
        print(outputs["overall_summary"])


def _prepare_soft_gate_parquet(
    *,
    source: Path,
    output_dir: Path,
    gate: str,
    score_column: str,
    rank_column: str,
    gate_threshold: float,
    temperature: float,
    rank_temperature: float,
    min_gate: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    if score_column not in frame.columns and gate != "rank_decay":
        raise ValueError(f"Score column {score_column!r} not found in {source}")
    if rank_column not in frame.columns and gate in {"rank_decay", "score_rank_product"}:
        raise ValueError(f"Rank column {rank_column!r} not found in {source}")
    frame[SOFT_WEIGHT_COLUMN] = _soft_gate_values(
        frame,
        gate=gate,
        score_column=score_column,
        rank_column=rank_column,
        gate_threshold=gate_threshold,
        temperature=temperature,
        rank_temperature=rank_temperature,
        min_gate=min_gate,
    )
    prepared = output_dir / "prepared_soft_gate_candidates.parquet"
    frame.to_parquet(prepared, index=False)
    return prepared


def _variant_name(name: str, suffix: str) -> str:
    suffix = suffix.strip("_")
    return name if not suffix else f"{name}_{suffix}"


def _method_key(method_key: str, suffix: str) -> str:
    suffix = suffix.strip("_")
    return method_key if not suffix else f"{method_key}_{suffix}"


def _method_label(method_label: str, suffix: str) -> str:
    suffix = suffix.strip("_")
    if not suffix:
        return method_label
    return f"{method_label} ({suffix})"


def _soft_gate_values(
    frame: pd.DataFrame,
    *,
    gate: str,
    score_column: str,
    rank_column: str,
    gate_threshold: float,
    temperature: float,
    rank_temperature: float,
    min_gate: float,
) -> pd.Series:
    if gate == "rank_decay":
        values = _rank_decay(frame[rank_column], rank_temperature=rank_temperature)
    elif gate == "score_rank_product":
        score = _clipped_score(frame[score_column], min_gate=min_gate)
        rank = _rank_decay(frame[rank_column], rank_temperature=rank_temperature)
        values = score * rank
    elif gate == "sigmoid_score":
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        score = pd.to_numeric(frame[score_column], errors="coerce").fillna(0.0)
        values = 1.0 / (1.0 + np.exp(-((score - gate_threshold) / temperature)))
    elif gate == "clipped_score":
        values = _clipped_score(frame[score_column], min_gate=min_gate)
    else:
        raise ValueError(f"Unknown soft gate: {gate}")
    return pd.Series(values, index=frame.index).clip(lower=min_gate, upper=1.0)


def _clipped_score(values: pd.Series, *, min_gate: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(min_gate)
    return numeric.clip(lower=min_gate, upper=1.0)


def _rank_decay(values: pd.Series, *, rank_temperature: float) -> pd.Series:
    if rank_temperature <= 0:
        raise ValueError("rank_temperature must be positive.")
    ranks = pd.to_numeric(values, errors="coerce").fillna(999.0)
    return np.exp(-((ranks - 1.0).clip(lower=0.0) / rank_temperature))


if __name__ == "__main__":
    main()
