#!/usr/bin/env python3
"""
run_qanet_20ep_eval.py
----------------------
Post-training evaluation pipeline for the 20-epoch QANet probe.

For every registered dataset/fold this script:
  1. Computes Spearman rank-correlation between qanet_score and true_iou
     (per split: train / validation / test).
  2. If test-split Spearman r >= MIN_TEST_R (default 0.40) for that fold,
     runs SIMPLE fusion pruned with source_balanced_topk + qanet_score.
  3. Writes per-fold fusion comparison CSVs and a summary table.

Outputs land in:
  hpc_results/soft_hybrid_probe/qanet_20ep_probe/
    qanet_20ep_rank_correlation.csv
    qanet_20ep_vs_pairwise_comparison.csv
    qanet_20ep_vs_pairwise_comparison.md

Usage:
    python scripts/run_qanet_20ep_eval.py
    python scripts/run_qanet_20ep_eval.py --datasets BF-C2DL-MuSC Fluo-C2DL-MSC
    python scripts/run_qanet_20ep_eval.py --skip-fusion
    python scripts/run_qanet_20ep_eval.py --min-test-r 0.0  # run fusion always
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Registry: one entry per (dataset, crop_sz, variant, fold)
# ---------------------------------------------------------------------------
PROBE_20EP_ROOT = Path("hpc_results/soft_hybrid_probe/qanet_20ep_probe")

FOLDS: list[dict] = [
    dict(
        dataset="BF-C2DL-HSC",
        crop_tag="sz64",
        variant="baseline",
        fold="fold-1",
        candidate_parquet="data/paper_runs/paper_inputs/BF-C2DL-HSC/sz64/baseline/fold-1_paper_ready.parquet",
    ),
    dict(
        dataset="BF-C2DL-HSC",
        crop_tag="sz64",
        variant="baseline",
        fold="fold-2",
        candidate_parquet="data/paper_runs/paper_inputs/BF-C2DL-HSC/sz64/baseline/fold-2_paper_ready.parquet",
    ),
    dict(
        dataset="BF-C2DL-MuSC",
        crop_tag="sz256",
        variant="musc_crop256",
        fold="fold-1",
        candidate_parquet="data/paper_runs/paper_inputs/BF-C2DL-MuSC/sz256/musc_crop256/fold-1_paper_ready.parquet",
    ),
    dict(
        dataset="BF-C2DL-MuSC",
        crop_tag="sz256",
        variant="musc_crop256",
        fold="fold-2",
        candidate_parquet="data/paper_runs/paper_inputs/BF-C2DL-MuSC/sz256/musc_crop256/fold-2_paper_ready.parquet",
    ),
    dict(
        dataset="DIC-C2DH-HeLa",
        crop_tag="sz256",
        variant="hela_crop256",
        fold="fold-1",
        candidate_parquet="hpc_results/hela/paper_runs/paper_inputs/DIC-C2DH-HeLa/sz256/hela_crop256/fold-1_paper_ready.parquet",
    ),
    dict(
        dataset="DIC-C2DH-HeLa",
        crop_tag="sz256",
        variant="hela_crop256",
        fold="fold-2",
        candidate_parquet="hpc_results/hela/paper_runs/paper_inputs/DIC-C2DH-HeLa/sz256/hela_crop256/fold-2_paper_ready.parquet",
    ),
    dict(
        dataset="Fluo-C2DL-MSC",
        crop_tag="sz1024",
        variant="msc_crop1024",
        fold="fold-1",
        candidate_parquet="hpc_results/msc/paper_runs/paper_inputs/Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-1_paper_ready.parquet",
    ),
    dict(
        dataset="Fluo-C2DL-MSC",
        crop_tag="sz1024",
        variant="msc_crop1024",
        fold="fold-2",
        candidate_parquet="hpc_results/msc/paper_runs/paper_inputs/Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-2_paper_ready.parquet",
    ),
    dict(
        dataset="Fluo-N2DH-GOWT1",
        crop_tag="sz160",
        variant="gowt1_crop160",
        fold="fold-1",
        candidate_parquet="hpc_results/gowt1/gowt1_sz160_reduced_20260514/paper_runs/paper_inputs/Fluo-N2DH-GOWT1/sz160/gowt1_crop160/fold-1_paper_ready.parquet",
    ),
    dict(
        dataset="Fluo-N2DH-GOWT1",
        crop_tag="sz160",
        variant="gowt1_crop160",
        fold="fold-2",
        candidate_parquet="hpc_results/gowt1/gowt1_sz160_reduced_20260514/paper_runs/paper_inputs/Fluo-N2DH-GOWT1/sz160/gowt1_crop160/fold-2_paper_ready.parquet",
    ),
]

PAIRWISE_BASELINE_CSV = Path(
    "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
    "all_available_simple_policy_comparison_rows.csv"
)
SOURCE_PRIOR_BASELINE_CSV = Path(
    "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
    "source_prior_topk5_simple/source_prior_topk5_simple_comparison_rows.csv"
)

FUSION_RUNNER = Path("scripts/run_ranker_pruned_fusion.py")
DEFAULT_MIN_TEST_R = 0.40


def scored_csv_path(repo: Path, entry: dict) -> Path:
    return (
        repo
        / PROBE_20EP_ROOT
        / entry["dataset"]
        / entry["crop_tag"]
        / entry["variant"]
        / entry["fold"]
        / "qanet_scored_candidates.csv"
    )


def fusion_output_dir(repo: Path, entry: dict) -> Path:
    return (
        repo
        / PROBE_20EP_ROOT
        / entry["dataset"]
        / entry["crop_tag"]
        / entry["variant"]
        / entry["fold"]
        / "qanet_20ep_source_balanced_top3_simple"
    )


# ---------------------------------------------------------------------------
# Step 1 – Rank correlation
# ---------------------------------------------------------------------------

def compute_rank_correlations(repo: Path, folds: list[dict]) -> pd.DataFrame:
    records = []
    for entry in folds:
        p = scored_csv_path(repo, entry)
        if not p.exists():
            print(f"  [WARN] scored CSV missing: {p}")
            continue
        df = pd.read_csv(p)
        for split in ("test", "validation", "train"):
            sub = df[df["split"] == split].dropna(subset=["qanet_score", "true_iou"])
            if len(sub) < 5:
                continue
            r, pval = spearmanr(sub["qanet_score"], sub["true_iou"])
            records.append(
                {
                    "dataset": entry["dataset"],
                    "fold": entry["fold"],
                    "split": split,
                    "spearman_r": float(r),
                    "p_value": float(pval),
                    "n": len(sub),
                }
            )
    return pd.DataFrame(records)


def print_rank_correlation_table(corr_df: pd.DataFrame) -> None:
    header = (
        f"{'Dataset':22s} {'Fold':6s} {'Split':12s} "
        f"{'Spearman r':12s} {'N':6s}"
    )
    print(header)
    print("-" * len(header))
    for _, row in corr_df.iterrows():
        gate = ""
        if row["split"] == "test" and float(row["spearman_r"]) >= DEFAULT_MIN_TEST_R:
            gate = " ✓"
        elif row["split"] == "test":
            gate = " ✗ BELOW GATE"
        print(
            f"{row['dataset']:22s} {row['fold']:6s} {row['split']:12s} "
            f"{float(row['spearman_r']):12.4f} {int(row['n']):6d}{gate}"
        )
    print()


# ---------------------------------------------------------------------------
# Step 2 – Fusion evaluation
# ---------------------------------------------------------------------------

def run_fusion(repo: Path, entry: dict, dry_run: bool = False) -> bool:
    label = f"{entry['dataset']}/{entry['fold']}"
    candidate_parquet = repo / entry["candidate_parquet"]
    scored_csv = scored_csv_path(repo, entry)
    out_dir = fusion_output_dir(repo, entry)

    if not candidate_parquet.exists():
        print(f"  [WARN] candidate parquet missing: {candidate_parquet}")
        return False
    if not scored_csv.exists():
        print(f"  [WARN] scored CSV missing: {scored_csv}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(repo / FUSION_RUNNER),
        "--candidate-parquet", str(candidate_parquet),
        "--ranked-candidates", str(scored_csv),
        "--output-dir", str(out_dir),
        "--policy", "source_balanced_topk",
        "--score-column", "qanet_score",
        "--top-k", "3",
        "--source-top-k", "1",
        "--models", "SIMPLE",
    ]
    print(f"  [FUSE] {label}")
    if dry_run:
        print("  " + " ".join(cmd))
        return True
    result = subprocess.run(cmd, cwd=repo, capture_output=False)
    if result.returncode != 0:
        print(f"  [FAIL] fusion exit code {result.returncode} for {label}", file=sys.stderr)
        return False
    print(f"  [OK]   {label}")
    return True


# ---------------------------------------------------------------------------
# Step 3 – Comparison table
# ---------------------------------------------------------------------------

def load_fusion_result(repo: Path, entry: dict) -> float | None:
    comparison_csv = fusion_output_dir(repo, entry) / "ranker_pruned_fusion_comparison.csv"
    if not comparison_csv.exists():
        return None
    df = pd.read_csv(comparison_csv)
    if "mean_iou" not in df.columns or df.empty:
        return None
    return float(df["mean_iou"].iloc[0])


def load_pairwise_baseline(datasets: list[str]) -> dict[tuple, float]:
    if not PAIRWISE_BASELINE_CSV.exists():
        print(f"[WARN] pairwise baseline CSV not found: {PAIRWISE_BASELINE_CSV}")
        return {}
    df = pd.read_csv(PAIRWISE_BASELINE_CSV)
    sub = df[(df["policy"] == "source_balanced_topk") & df["dataset"].isin(datasets)]
    return {(row["dataset"], row["fold"]): row["mean_iou"] for _, row in sub.iterrows()}


def load_source_prior_baseline(datasets: list[str]) -> dict[tuple, float]:
    if not SOURCE_PRIOR_BASELINE_CSV.exists():
        print(f"[WARN] source-prior baseline CSV not found: {SOURCE_PRIOR_BASELINE_CSV}")
        return {}
    df = pd.read_csv(SOURCE_PRIOR_BASELINE_CSV)
    sub = df[df["dataset"].isin(datasets)]
    return {(row["dataset"], row["fold"]): row["mean_iou"] for _, row in sub.iterrows()}


def build_comparison_table(
    repo: Path,
    folds: list[dict],
    corr_df: pd.DataFrame,
    min_test_r: float,
) -> pd.DataFrame:
    datasets = sorted({f["dataset"] for f in folds})
    pw_lookup = load_pairwise_baseline(datasets)
    sp_lookup = load_source_prior_baseline(datasets)

    # Index test-split r by (dataset, fold)
    r_lookup: dict[tuple, float] = {}
    for _, row in corr_df[corr_df["split"] == "test"].iterrows():
        r_lookup[(row["dataset"], row["fold"])] = float(row["spearman_r"])

    records = []
    for entry in folds:
        key = (entry["dataset"], entry["fold"])
        qanet_iou = load_fusion_result(repo, entry)
        pairwise_iou = pw_lookup.get(key, float("nan"))
        sp_iou = sp_lookup.get(key, float("nan"))
        test_r = r_lookup.get(key, float("nan"))

        records.append(
            {
                "dataset": entry["dataset"],
                "fold": entry["fold"],
                "test_spearman_r": test_r,
                "gate_passed": test_r >= min_test_r,
                "pairwise_lr_top3": pairwise_iou,
                "source_prior_top5": sp_iou,
                "qanet_20ep_top3": qanet_iou if qanet_iou is not None else float("nan"),
                "delta_qanet_vs_pairwise": (
                    (qanet_iou - pairwise_iou)
                    if qanet_iou is not None
                    else float("nan")
                ),
                "delta_qanet_vs_source_prior": (
                    (qanet_iou - sp_iou)
                    if qanet_iou is not None
                    else float("nan")
                ),
            }
        )

    fold_df = pd.DataFrame(records)

    # Add dataset mean rows
    mean_rows = []
    for dataset, grp in fold_df.groupby("dataset"):
        mean_rows.append(
            {
                "dataset": dataset,
                "fold": "Mean",
                "test_spearman_r": grp["test_spearman_r"].mean(),
                "gate_passed": grp["gate_passed"].all(),
                "pairwise_lr_top3": grp["pairwise_lr_top3"].mean(),
                "source_prior_top5": grp["source_prior_top5"].mean(),
                "qanet_20ep_top3": grp["qanet_20ep_top3"].mean(),
                "delta_qanet_vs_pairwise": grp["delta_qanet_vs_pairwise"].mean(),
                "delta_qanet_vs_source_prior": grp["delta_qanet_vs_source_prior"].mean(),
            }
        )

    all_rows = pd.concat([fold_df, pd.DataFrame(mean_rows)], ignore_index=True)
    all_rows["_sort_fold"] = all_rows["fold"].apply(lambda x: "z" + x if x == "Mean" else x)
    all_rows = all_rows.sort_values(["dataset", "_sort_fold"]).drop(columns=["_sort_fold"])
    return all_rows.reset_index(drop=True)


def render_markdown(table: pd.DataFrame, min_test_r: float) -> str:
    date = "2026-05-29"
    lines = [
        "# QANet 20-Epoch Probe vs Tabular Rankers: Full-Dataset Comparison",
        "",
        f"Date: {date}",
        "",
        "## Scope",
        "",
        "This table compares the QANet 20-epoch probe (20 epochs, 64 px crops,",
        "source identity enabled, auto device) against two tabular baselines",
        "on all five CTC datasets under the strict fold protocol.",
        "",
        f"Gate criterion: test-split Spearman r >= {min_test_r:.2f}.",
        "Folds below the gate had fusion skipped; IoU is n/a.",
        "",
        "- **Pairwise LR top3**: `pairwise_cluster_score`, top_k=3, source_top_k=1, SIMPLE.",
        "- **Source-prior top5**: `source_train_mean_iou`, top_k=5, source_top_k=1, SIMPLE.",
        "- **QANet 20ep top3**: `qanet_score` (20-epoch CNN), top_k=3, source_top_k=1, SIMPLE.",
        "",
        "## Results",
        "",
        "| Dataset | Fold | Test r | Gate | Pairwise LR top3 | Source-prior top5 | QANet 20ep top3 | Δ (QANet vs LR) |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|",
    ]

    def fmt(val: object) -> str:
        if pd.isna(val):
            return "n/a"
        return f"{float(val):.6f}"

    def fmtd(val: object) -> str:
        if pd.isna(val):
            return "n/a"
        return f"{float(val):+.6f}"

    def fmtr(val: object) -> str:
        if pd.isna(val):
            return "n/a"
        return f"{float(val):.4f}"

    for _, row in table.iterrows():
        fold_label = f"**{row['fold']}**" if row["fold"] == "Mean" else row["fold"]
        gate_sym = "✓" if row["gate_passed"] else "✗"
        lines.append(
            f"| {row['dataset']} | {fold_label}"
            f" | {fmtr(row['test_spearman_r'])}"
            f" | {gate_sym}"
            f" | {fmt(row['pairwise_lr_top3'])}"
            f" | {fmt(row['source_prior_top5'])}"
            f" | {fmt(row['qanet_20ep_top3'])}"
            f" | {fmtd(row['delta_qanet_vs_pairwise'])}"
            f" |"
        )

    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- Folds with test Spearman r below the gate criterion have fusion results that",
        "  would not be trustworthy. These rows show n/a for QANet 20ep IoU.",
        "- Positive Δ means QANet 20ep pruning improved over fixed pairwise top3.",
        "- A properly-trained QANet (r >= 0.40 test) must beat BOTH pairwise LR AND",
        "  source-prior top5 on the target datasets to be considered an improvement.",
        "- Do not use these results as paper rows without reviewing this table fully.",
        "",
        "## Files",
        "",
        f"Rank correlation: `{PROBE_20EP_ROOT}/qanet_20ep_rank_correlation.csv`",
        f"This comparison: `{PROBE_20EP_ROOT}/qanet_20ep_vs_pairwise_comparison.csv`",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="DS",
        help="Restrict to these datasets (default: all).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        metavar="FOLD",
        help="Restrict to these folds (default: all).",
    )
    parser.add_argument(
        "--min-test-r",
        type=float,
        default=DEFAULT_MIN_TEST_R,
        help=f"Minimum test-split Spearman r to run fusion (default {DEFAULT_MIN_TEST_R}).",
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Compute rank correlations only, skip fusion runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print fusion commands but do not run them.",
    )
    parser.add_argument(
        "--force-fusion",
        action="store_true",
        help="Run fusion even if fusion output already exists.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    folds = FOLDS
    if args.datasets:
        folds = [f for f in folds if f["dataset"] in args.datasets]
    if args.folds:
        folds = [f for f in folds if f["fold"] in args.folds]

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Rank correlation (qanet_score vs true_iou)")
    print("=" * 60)
    corr_df = compute_rank_correlations(repo, folds)

    if corr_df.empty:
        print("[ERROR] No scored candidates found. Run train_qanet_all_folds.py first.")
        sys.exit(1)

    print_rank_correlation_table(corr_df)

    out_root = repo / PROBE_20EP_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    corr_csv = out_root / "qanet_20ep_rank_correlation.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"Wrote rank correlation CSV: {corr_csv}")

    # ------------------------------------------------------------------
    if args.skip_fusion:
        print("\n[INFO] --skip-fusion set; skipping fusion step.")
        return

    print()
    print("=" * 60)
    print(f"Step 2: Fusion (gate: test Spearman r >= {args.min_test_r:.2f})")
    print("=" * 60)

    # Build test-r lookup
    r_lookup: dict[tuple, float] = {}
    for _, row in corr_df[corr_df["split"] == "test"].iterrows():
        r_lookup[(row["dataset"], row["fold"])] = float(row["spearman_r"])

    for entry in folds:
        label = f"{entry['dataset']}/{entry['fold']}"
        key = (entry["dataset"], entry["fold"])
        test_r = r_lookup.get(key, float("nan"))

        if test_r < args.min_test_r:
            print(
                f"  [SKIP] {label} — test Spearman r={test_r:.4f} "
                f"below gate {args.min_test_r:.2f}"
            )
            continue

        out_dir = fusion_output_dir(repo, entry)
        comparison_csv = out_dir / "ranker_pruned_fusion_comparison.csv"
        if not args.force_fusion and comparison_csv.exists():
            print(f"  [DONE] {label} — fusion result already exists, skipping.")
            continue

        run_fusion(repo, entry, dry_run=args.dry_run)

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Building comparison table")
    print("=" * 60)
    table = build_comparison_table(repo, folds, corr_df, args.min_test_r)

    comparison_csv = out_root / "qanet_20ep_vs_pairwise_comparison.csv"
    comparison_md = out_root / "qanet_20ep_vs_pairwise_comparison.md"
    table.to_csv(comparison_csv, index=False)
    comparison_md.write_text(render_markdown(table, args.min_test_r))

    print(f"Wrote comparison CSV:      {comparison_csv}")
    print(f"Wrote comparison markdown: {comparison_md}")
    print()

    display_cols = [
        "dataset", "fold", "test_spearman_r",
        "pairwise_lr_top3", "qanet_20ep_top3", "delta_qanet_vs_pairwise",
    ]
    print(table[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
