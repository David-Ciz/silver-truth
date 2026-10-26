#!/usr/bin/env python3
"""
run_ranker_benchmark.py
-----------------------
Registry-driven runner: reads configs/ranker_benchmark.yaml and runs
run_ranker_pruned_fusion.py for every registered method × fold combination.

Outputs land in:
    hpc_results/ranker_benchmark/{method_name}/{dataset}/{variant}/{fold}/

Already-completed runs are skipped (idempotent).

Usage:
    # Run everything (skips already-done)
    python scripts/run_ranker_benchmark.py

    # Run only specific methods
    python scripts/run_ranker_benchmark.py --methods pairwise_lr mlp_tabular

    # Run only specific folds
    python scripts/run_ranker_benchmark.py --folds "BF-C2DL-MuSC/fold-1"

    # Dry-run (print commands, don't execute)
    python scripts/run_ranker_benchmark.py --dry-run

    # Force re-run even if output already exists
    python scripts/run_ranker_benchmark.py --force
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def output_dir_for(root: Path, method_name: str, fold_def: dict) -> Path:
    return (
        root
        / method_name
        / fold_def["dataset"]
        / fold_def["variant"]
        / fold_def["fold"]
    )


def is_done(out_dir: Path) -> bool:
    return (out_dir / "ranker_pruned_fusion_comparison.csv").exists()


def build_command(
    runner: Path,
    candidate_parquet: Path,
    ranked_candidates: Path,
    out_dir: Path,
    method: dict,
    benchmark: dict,
) -> list[str]:
    top_k = str(method.get("top_k", benchmark.get("top_k", 3)))
    source_top_k = str(method.get("source_top_k", benchmark.get("source_top_k", 1)))
    policy = method.get("policy", benchmark.get("policy", "source_balanced_topk"))
    models_list = method.get("models", benchmark.get("models", ["SIMPLE"]))
    # models is a list, but the CLI takes one value at a time; most runs use SIMPLE
    models = models_list[0] if isinstance(models_list, list) else models_list

    cmd = [
        sys.executable,
        str(runner),
        "--candidate-parquet", str(candidate_parquet),
        "--ranked-candidates", str(ranked_candidates),
        "--output-dir", str(out_dir),
        "--policy", policy,
        "--score-column", method["score_column"],
        "--top-k", top_k,
        "--source-top-k", source_top_k,
        "--models", models,
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Registry-driven ranker benchmark runner")
    parser.add_argument(
        "--config",
        default="configs/ranker_benchmark.yaml",
        help="Path to registry YAML (default: configs/ranker_benchmark.yaml)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        metavar="METHOD",
        help="Only run these method names (default: all)",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        metavar="FOLD_ID",
        help='Only run these fold IDs, e.g. "BF-C2DL-MuSC/fold-1" (default: all)',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / args.config
    cfg = load_config(config_path)

    benchmark = cfg["benchmark"]
    output_root = repo_root / benchmark["output_root"]
    runner = repo_root / "scripts" / "run_ranker_pruned_fusion.py"

    folds_by_id = {f["id"]: f for f in cfg["folds"]}
    methods = cfg["methods"]

    # Apply filters
    if args.methods:
        methods = [m for m in methods if m["name"] in args.methods]
        if not methods:
            print(f"[ERROR] No methods matched filter: {args.methods}", file=sys.stderr)
            sys.exit(1)

    if args.folds:
        unknown = set(args.folds) - set(folds_by_id)
        if unknown:
            print(f"[ERROR] Unknown fold IDs: {unknown}", file=sys.stderr)
            sys.exit(1)
        folds_by_id = {k: v for k, v in folds_by_id.items() if k in args.folds}

    total = 0
    skipped = 0
    ran = 0
    failed = 0

    for method in methods:
        m_name = method["name"]
        scored = method.get("scored_candidates", {})

        for fold_id, fold_def in folds_by_id.items():
            total += 1

            if fold_id not in scored:
                print(f"[SKIP]  {m_name} × {fold_id}  — no scored_candidates entry")
                skipped += 1
                continue

            candidate_parquet = repo_root / fold_def["candidate_parquet"]
            ranked_candidates = repo_root / scored[fold_id]
            out_dir = output_dir_for(output_root, m_name, fold_def)

            if not args.force and is_done(out_dir):
                print(f"[DONE]  {m_name} × {fold_id}  — already exists, skipping")
                skipped += 1
                continue

            if not candidate_parquet.exists():
                print(f"[WARN]  {m_name} × {fold_id}  — candidate_parquet not found: {candidate_parquet}")
                skipped += 1
                continue

            if not ranked_candidates.exists():
                print(f"[WARN]  {m_name} × {fold_id}  — scored_candidates not found: {ranked_candidates}")
                skipped += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_command(runner, candidate_parquet, ranked_candidates, out_dir, method, benchmark)

            print(f"\n[RUN]   {m_name} × {fold_id}")
            print("  " + " ".join(cmd))

            if not args.dry_run:
                result = subprocess.run(cmd, cwd=repo_root)
                if result.returncode != 0:
                    print(f"[FAIL]  {m_name} × {fold_id}  — exit code {result.returncode}", file=sys.stderr)
                    failed += 1
                else:
                    print(f"[OK]    {m_name} × {fold_id}")
                    ran += 1
            else:
                ran += 1  # count as "would run" in dry-run mode

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Summary: {total} total, {ran} ran, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
