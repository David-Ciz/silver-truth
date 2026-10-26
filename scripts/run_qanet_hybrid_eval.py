#!/usr/bin/env python3
"""
run_qanet_hybrid_eval.py
------------------------
Combine QANet 20-epoch scores with pairwise cluster scores (hybrid scoring).

For each registered dataset/fold, this script:
  1. Loads the existing qanet_scored_candidates.csv (has qanet_score,
     qanet_rank_in_cluster, candidate_id, split, true_iou).
  2. Scores the paper_inputs candidate parquet with the pre-trained pairwise
     cluster ranker model (all_available probe) to get pairwise_cluster_score
     and pairwise_cluster_rank.
  3. Merges on candidate_id / cell_id.
  4. Computes hybrid_rank_score for each alpha in HYBRID_ALPHAS:
       hybrid = alpha * qanet_rank_in_cluster + (1-alpha) * pairwise_cluster_rank
  5. For folds that pass the test-Spearman gate (qanet_score vs true_iou
     >= MIN_TEST_R), runs SIMPLE fusion for each alpha variant.
  6. Builds a comparison table: per-fold and per-dataset mean IoU for every
     alpha, plus the pairwise-LR-only baseline (alpha=0) and the
     QANet-only baseline (alpha=1).

Gate criterion: test-split Spearman r of qanet_score vs true_iou >= 0.40.
Folds below the gate are excluded from fusion (alpha=0 pairwise-only result
still included as the baseline reference row).

Outputs land in:
  hpc_results/soft_hybrid_probe/qanet_hybrid_probe/
    qanet_hybrid_rank_correlation.csv
    qanet_hybrid_comparison.csv
    qanet_hybrid_comparison.md

Usage:
    python scripts/run_qanet_hybrid_eval.py
    python scripts/run_qanet_hybrid_eval.py --datasets BF-C2DL-MuSC Fluo-C2DL-MSC
    python scripts/run_qanet_hybrid_eval.py --skip-fusion
    python scripts/run_qanet_hybrid_eval.py --min-test-r 0.0  # run all
    python scripts/run_qanet_hybrid_eval.py --force-fusion
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Registry – one entry per (dataset, fold)
# ---------------------------------------------------------------------------

PROBE_20EP_ROOT = Path("hpc_results/soft_hybrid_probe/qanet_20ep_probe")
HYBRID_PROBE_ROOT = Path("hpc_results/soft_hybrid_probe/qanet_hybrid_probe")

# Pre-scored pairwise parquets from the broadest all-available probe (all 5 datasets,
# both folds, strict train-split protocol). These already have pairwise_cluster_score
# and pairwise_cluster_rank computed so we avoid re-running the expensive feature
# extraction pipeline.
PAIRWISE_SCORED_ROOT = Path(
    "hpc_results/soft_hybrid_probe/"
    "all_available_square_rect_full_image_local_support_pairwise_source_trainonly"
    "/features"
)

# alpha=0.0 => pure pairwise rank; alpha=1.0 => pure QANet rank
HYBRID_ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]

DEFAULT_MIN_TEST_R = 0.40

FUSION_RUNNER = Path("scripts/run_ranker_pruned_fusion.py")

PAIRWISE_BASELINE_CSV = Path(
    "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
    "all_available_simple_policy_comparison_rows.csv"
)
SOURCE_PRIOR_BASELINE_CSV = Path(
    "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
    "source_prior_topk5_simple/source_prior_topk5_simple_comparison_rows.csv"
)

FOLDS: list[dict] = [
    dict(
        dataset="BF-C2DL-HSC",
        crop_tag="sz64",
        variant="baseline",
        fold="fold-1",
        candidate_parquet="data/paper_runs/paper_inputs/BF-C2DL-HSC/sz64/baseline/fold-1_paper_ready.parquet",
        pairwise_ranked="BF-C2DL-HSC/sz64/baseline/fold-1_paper_ready_pairwise_ranked.parquet",
    ),
    dict(
        dataset="BF-C2DL-HSC",
        crop_tag="sz64",
        variant="baseline",
        fold="fold-2",
        candidate_parquet="data/paper_runs/paper_inputs/BF-C2DL-HSC/sz64/baseline/fold-2_paper_ready.parquet",
        pairwise_ranked="BF-C2DL-HSC/sz64/baseline/fold-2_paper_ready_pairwise_ranked.parquet",
    ),
    dict(
        dataset="BF-C2DL-MuSC",
        crop_tag="sz256",
        variant="musc_crop256",
        fold="fold-1",
        candidate_parquet="data/paper_runs/paper_inputs/BF-C2DL-MuSC/sz256/musc_crop256/fold-1_paper_ready.parquet",
        pairwise_ranked="BF-C2DL-MuSC/sz256/musc_crop256/fold-1_paper_ready_pairwise_ranked.parquet",
    ),
    dict(
        dataset="BF-C2DL-MuSC",
        crop_tag="sz256",
        variant="musc_crop256",
        fold="fold-2",
        candidate_parquet="data/paper_runs/paper_inputs/BF-C2DL-MuSC/sz256/musc_crop256/fold-2_paper_ready.parquet",
        pairwise_ranked="BF-C2DL-MuSC/sz256/musc_crop256/fold-2_paper_ready_pairwise_ranked.parquet",
    ),
    dict(
        dataset="DIC-C2DH-HeLa",
        crop_tag="sz256",
        variant="hela_crop256",
        fold="fold-1",
        candidate_parquet="hpc_results/hela/paper_runs/paper_inputs/DIC-C2DH-HeLa/sz256/hela_crop256/fold-1_paper_ready.parquet",
        pairwise_ranked="DIC-C2DH-HeLa/sz256/hela_crop256/fold-1_paper_ready_pairwise_ranked.parquet",
    ),
    dict(
        dataset="DIC-C2DH-HeLa",
        crop_tag="sz256",
        variant="hela_crop256",
        fold="fold-2",
        candidate_parquet="hpc_results/hela/paper_runs/paper_inputs/DIC-C2DH-HeLa/sz256/hela_crop256/fold-2_paper_ready.parquet",
        pairwise_ranked="DIC-C2DH-HeLa/sz256/hela_crop256/fold-2_paper_ready_pairwise_ranked.parquet",
    ),
    dict(
        dataset="Fluo-C2DL-MSC",
        crop_tag="sz1024",
        variant="msc_crop1024",
        fold="fold-1",
        candidate_parquet="hpc_results/msc/paper_runs/paper_inputs/Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-1_paper_ready.parquet",
        pairwise_ranked="Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-1_paper_ready_pairwise_ranked.parquet",
    ),
    dict(
        dataset="Fluo-C2DL-MSC",
        crop_tag="sz1024",
        variant="msc_crop1024",
        fold="fold-2",
        candidate_parquet="hpc_results/msc/paper_runs/paper_inputs/Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-2_paper_ready.parquet",
        pairwise_ranked="Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-2_paper_ready_pairwise_ranked.parquet",
    ),
    dict(
        dataset="Fluo-N2DH-GOWT1",
        crop_tag="sz160",
        variant="gowt1_crop160",
        fold="fold-1",
        candidate_parquet="hpc_results/gowt1/gowt1_sz160_reduced_20260514/paper_runs/paper_inputs/Fluo-N2DH-GOWT1/sz160/gowt1_crop160/fold-1_paper_ready.parquet",
        pairwise_ranked="Fluo-N2DH-GOWT1/sz160/gowt1_crop160/fold-1_paper_ready_pairwise_ranked.parquet",
    ),
    dict(
        dataset="Fluo-N2DH-GOWT1",
        crop_tag="sz160",
        variant="gowt1_crop160",
        fold="fold-2",
        candidate_parquet="hpc_results/gowt1/gowt1_sz160_reduced_20260514/paper_runs/paper_inputs/Fluo-N2DH-GOWT1/sz160/gowt1_crop160/fold-2_paper_ready.parquet",
        pairwise_ranked="Fluo-N2DH-GOWT1/sz160/gowt1_crop160/fold-2_paper_ready_pairwise_ranked.parquet",
    ),
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def qanet_scored_csv(repo: Path, entry: dict) -> Path:
    return (
        repo / PROBE_20EP_ROOT
        / entry["dataset"] / entry["crop_tag"] / entry["variant"]
        / entry["fold"] / "qanet_scored_candidates.csv"
    )


def pairwise_ranked_path(repo: Path, entry: dict) -> Path:
    return repo / PAIRWISE_SCORED_ROOT / entry["pairwise_ranked"]


def hybrid_fold_dir(repo: Path, entry: dict) -> Path:
    return (
        repo / HYBRID_PROBE_ROOT
        / entry["dataset"] / entry["crop_tag"] / entry["variant"]
        / entry["fold"]
    )


def hybrid_scored_csv(repo: Path, entry: dict) -> Path:
    return hybrid_fold_dir(repo, entry) / "hybrid_scored_candidates.csv"


def hybrid_fusion_dir(repo: Path, entry: dict, alpha: float) -> Path:
    alpha_tag = f"alpha{alpha:.2f}".replace(".", "p")
    return hybrid_fold_dir(repo, entry) / f"hybrid_{alpha_tag}_source_balanced_top3_simple"


# ---------------------------------------------------------------------------
# Step 1: Build hybrid scored CSV
# ---------------------------------------------------------------------------

def build_hybrid_scored_csv(repo: Path, entry: dict) -> pd.DataFrame | None:
    """
    Merge QANet scores with pre-scored pairwise cluster scores and compute
    hybrid_rank scores for each alpha in HYBRID_ALPHAS.

    Strategy: rank-based combination within each cluster:
      hybrid_rank = alpha * qanet_rank_in_cluster + (1-alpha) * pairwise_cluster_rank
    Both rank columns are percentile ranks in [0,1] (higher = better).
    The combined hybrid_rank is then used as the score column for pruning.

    The pairwise_cluster_rank is read from the pre-scored
    all_available_square_rect_full_image_local_support_pairwise_source_trainonly
    parquets which already have the full feature set and pairwise scores computed.
    """
    label = f"{entry['dataset']}/{entry['fold']}"

    # 1. Load QANet scored CSV
    qanet_csv = qanet_scored_csv(repo, entry)
    if not qanet_csv.exists():
        print(f"  [WARN] QANet scored CSV missing: {qanet_csv}")
        return None
    qanet_df = pd.read_csv(qanet_csv)
    if "qanet_rank_in_cluster" not in qanet_df.columns:
        print(f"  [WARN] qanet_rank_in_cluster not in {qanet_csv.name}")
        return None
    if "candidate_id" not in qanet_df.columns:
        print(f"  [WARN] candidate_id not in {qanet_csv.name}")
        return None

    # 2. Load pre-scored pairwise parquet (has pairwise_cluster_score,
    #    pairwise_cluster_rank, cell_id)
    pw_parquet = pairwise_ranked_path(repo, entry)
    if not pw_parquet.exists():
        print(f"  [WARN] pairwise ranked parquet missing: {pw_parquet}")
        return None
    pw_df = pd.read_parquet(pw_parquet)

    for col in ("pairwise_cluster_score", "pairwise_cluster_rank", "candidate_id"):
        if col not in pw_df.columns:
            print(f"  [ERROR] column {col!r} not in pairwise ranked parquet for {label}")
            return None

    # Build lookup: candidate_id → pairwise columns
    pw_lookup = pw_df[["candidate_id", "pairwise_cluster_score", "pairwise_cluster_rank"]].copy()
    pw_lookup = pw_lookup.drop_duplicates("candidate_id")

    merged = qanet_df.merge(
        pw_lookup,
        on="candidate_id",
        how="left",
        suffixes=("", "_pw"),
    )
    n_matched = merged["pairwise_cluster_rank"].notna().sum()
    n_total = len(merged)
    if n_matched == 0:
        print(f"  [ERROR] 0/{n_total} candidate_ids matched pairwise scores for {label}")
        return None
    if n_matched < n_total:
        miss = n_total - n_matched
        print(f"  [WARN]  {miss}/{n_total} unmatched — filling with cluster mean or 0.5")
        merged["pairwise_cluster_rank"] = merged["pairwise_cluster_rank"].fillna(0.5)
        merged["pairwise_cluster_score"] = merged["pairwise_cluster_score"].fillna(0.5)

    # 3. Compute hybrid_rank for each alpha
    for alpha in HYBRID_ALPHAS:
        col = _alpha_col(alpha)
        qanet_rank = merged["qanet_rank_in_cluster"].fillna(0.5)
        pw_rank = merged["pairwise_cluster_rank"].fillna(0.5)
        merged[col] = (alpha * qanet_rank + (1.0 - alpha) * pw_rank).clip(0.0, 1.0)

    # 4. Write hybrid scored CSV
    out_dir = hybrid_fold_dir(repo, entry)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = hybrid_scored_csv(repo, entry)
    merged.to_csv(out_csv, index=False)
    print(f"  [HYBRID] {label} — wrote {len(merged)} rows to {out_csv.name} "
          f"({n_matched}/{n_total} pairwise matches)")
    return merged


def _alpha_col(alpha: float) -> str:
    return f"hybrid_rank_alpha{alpha:.2f}".replace(".", "p")


# ---------------------------------------------------------------------------
# Step 2: Rank correlation
# ---------------------------------------------------------------------------

def compute_rank_correlations(repo: Path, folds: list[dict]) -> pd.DataFrame:
    """Re-use rank correlation from existing QANet 20ep CSV if available."""
    existing_csv = repo / HYBRID_PROBE_ROOT / "qanet_hybrid_rank_correlation.csv"
    # Try to load from QANet 20ep probe first
    qanet_corr_csv = repo / PROBE_20EP_ROOT / "qanet_20ep_rank_correlation.csv"
    if qanet_corr_csv.exists():
        return pd.read_csv(qanet_corr_csv)

    records = []
    for entry in folds:
        hcsv = hybrid_scored_csv(repo, entry)
        if not hcsv.exists():
            continue
        df = pd.read_csv(hcsv)
        for split in ("test", "validation", "train"):
            sub = df[df["split"] == split].dropna(subset=["qanet_score", "true_iou"])
            if len(sub) < 5:
                continue
            r, pval = spearmanr(sub["qanet_score"], sub["true_iou"])
            records.append({
                "dataset": entry["dataset"],
                "fold": entry["fold"],
                "split": split,
                "spearman_r": float(r),
                "n": len(sub),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 3: Fusion for each alpha
# ---------------------------------------------------------------------------

def run_fusion_alpha(
    repo: Path,
    entry: dict,
    alpha: float,
    dry_run: bool = False,
) -> bool:
    label = f"{entry['dataset']}/{entry['fold']} alpha={alpha:.2f}"
    cand_parquet = repo / entry["candidate_parquet"]
    scored_csv = hybrid_scored_csv(repo, entry)
    out_dir = hybrid_fusion_dir(repo, entry, alpha)
    score_col = _alpha_col(alpha)

    if not cand_parquet.exists():
        print(f"  [WARN] candidate parquet missing: {cand_parquet}")
        return False
    if not scored_csv.exists():
        print(f"  [WARN] hybrid scored CSV missing: {scored_csv}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(repo / FUSION_RUNNER),
        "--candidate-parquet", str(cand_parquet),
        "--ranked-candidates", str(scored_csv),
        "--output-dir", str(out_dir),
        "--policy", "source_balanced_topk",
        "--score-column", score_col,
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


def load_fusion_iou(repo: Path, entry: dict, alpha: float) -> float | None:
    comparison_csv = hybrid_fusion_dir(repo, entry, alpha) / "ranker_pruned_fusion_comparison.csv"
    if not comparison_csv.exists():
        return None
    df = pd.read_csv(comparison_csv)
    if "mean_iou" not in df.columns or df.empty:
        return None
    return float(df["mean_iou"].iloc[0])


# ---------------------------------------------------------------------------
# Step 4: Comparison table
# ---------------------------------------------------------------------------

def load_pairwise_baseline(datasets: list[str]) -> dict[tuple, float]:
    if not PAIRWISE_BASELINE_CSV.exists():
        print(f"[WARN] pairwise baseline CSV not found: {PAIRWISE_BASELINE_CSV}")
        return {}
    df = pd.read_csv(PAIRWISE_BASELINE_CSV)
    sub = df[(df["policy"] == "source_balanced_topk") & df["dataset"].isin(datasets)]
    return {(row["dataset"], row["fold"]): row["mean_iou"] for _, row in sub.iterrows()}


def load_qanet_20ep_iou(repo: Path, folds: list[dict]) -> dict[tuple, float]:
    """Load QANet-only fusion IoU from the 20ep probe."""
    result: dict[tuple, float] = {}
    for entry in folds:
        fusion_dir = (
            repo / PROBE_20EP_ROOT
            / entry["dataset"] / entry["crop_tag"] / entry["variant"]
            / entry["fold"] / "qanet_20ep_source_balanced_top3_simple"
        )
        comparison_csv = fusion_dir / "ranker_pruned_fusion_comparison.csv"
        if comparison_csv.exists():
            df = pd.read_csv(comparison_csv)
            if "mean_iou" in df.columns and not df.empty:
                result[(entry["dataset"], entry["fold"])] = float(df["mean_iou"].iloc[0])
    return result


def build_comparison_table(
    repo: Path,
    folds: list[dict],
    corr_df: pd.DataFrame,
    min_test_r: float,
) -> pd.DataFrame:
    datasets = sorted({f["dataset"] for f in folds})
    pw_lookup = load_pairwise_baseline(datasets)
    qanet_lookup = load_qanet_20ep_iou(repo, folds)

    # Index test-split r by (dataset, fold)
    r_lookup: dict[tuple, float] = {}
    for _, row in corr_df[corr_df["split"] == "test"].iterrows():
        r_lookup[(row["dataset"], row["fold"])] = float(row["spearman_r"])

    records = []
    for entry in folds:
        key = (entry["dataset"], entry["fold"])
        test_r = r_lookup.get(key, float("nan"))
        gate_passed = test_r >= min_test_r

        row: dict = {
            "dataset": entry["dataset"],
            "fold": entry["fold"],
            "test_spearman_r": test_r,
            "gate_passed": gate_passed,
            "pairwise_lr_top3": pw_lookup.get(key, float("nan")),
            "qanet_20ep_top3": qanet_lookup.get(key, float("nan")),
        }

        for alpha in HYBRID_ALPHAS:
            col = f"hybrid_alpha_{alpha:.2f}".replace(".", "p")
            if gate_passed or alpha == 0.0:
                iou = load_fusion_iou(repo, entry, alpha)
                row[col] = iou if iou is not None else float("nan")
            else:
                row[col] = float("nan")

        records.append(row)

    fold_df = pd.DataFrame(records)

    # Dataset mean rows
    mean_rows = []
    for dataset, grp in fold_df.groupby("dataset"):
        mean_row: dict = {
            "dataset": dataset,
            "fold": "Mean",
            "test_spearman_r": grp["test_spearman_r"].mean(),
            "gate_passed": grp["gate_passed"].all(),
            "pairwise_lr_top3": grp["pairwise_lr_top3"].mean(),
            "qanet_20ep_top3": grp["qanet_20ep_top3"].mean(),
        }
        for alpha in HYBRID_ALPHAS:
            col = f"hybrid_alpha_{alpha:.2f}".replace(".", "p")
            mean_row[col] = grp[col].mean()
        mean_rows.append(mean_row)

    all_rows = pd.concat([fold_df, pd.DataFrame(mean_rows)], ignore_index=True)
    all_rows["_sort"] = all_rows["fold"].apply(lambda x: "z" + x if x == "Mean" else x)
    all_rows = all_rows.sort_values(["dataset", "_sort"]).drop(columns=["_sort"])
    return all_rows.reset_index(drop=True)


def render_markdown(table: pd.DataFrame, min_test_r: float) -> str:
    alpha_cols = [
        (alpha, f"hybrid_alpha_{alpha:.2f}".replace(".", "p"))
        for alpha in HYBRID_ALPHAS
    ]

    def fmt(val: object) -> str:
        if pd.isna(val):
            return "n/a"
        return f"{float(val):.6f}"

    def fmtd(val: object, ref: object) -> str:
        if pd.isna(val) or pd.isna(ref):
            return "n/a"
        delta = float(val) - float(ref)
        return f"{delta:+.6f}"

    def fmtr(val: object) -> str:
        if pd.isna(val):
            return "n/a"
        return f"{float(val):.4f}"

    alpha_headers = " | ".join(f"Hybrid α={a:.2f}" for a, _ in alpha_cols)
    alpha_seps = " | ".join("---:" for _ in alpha_cols)
    delta_headers = " | ".join(f"Δ α={a:.2f} vs LR" for a, _ in alpha_cols)
    delta_seps = " | ".join("---:" for _ in alpha_cols)

    lines = [
        "# QANet–Pairwise Hybrid Score: Full-Dataset Comparison",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This table evaluates combining QANet 20ep scores with pairwise cluster",
        "ranker scores using a rank-based hybrid formula:",
        "",
        "```",
        "hybrid_rank = alpha * qanet_rank_in_cluster + (1-alpha) * pairwise_cluster_rank",
        "```",
        "",
        "Both `qanet_rank_in_cluster` and `pairwise_cluster_rank` are percentile",
        "ranks within each candidate cluster (0 = worst, 1 = best in cluster).",
        "The hybrid rank is used as the score column for source_balanced_topk,",
        "top_k=3, source_top_k=1, SIMPLE fusion.",
        "",
        f"Gate: test-split Spearman r (qanet_score vs true_iou) >= {min_test_r:.2f}.",
        "Folds below the gate have n/a for QANet-containing alphas.",
        "α=0.00 (pairwise-only) is always included as the baseline.",
        "",
        "**Reporting guardrail**: these are diagnostic rows. Do not promote a",
        "hybrid result as paper-ready unless it improves consistently over the",
        "fixed pairwise top3 baseline on all datasets under the strict protocol.",
        "",
        "## Score Sources",
        "",
        "- **Pairwise LR** (external baseline): `pairwise_cluster_score`, same models",
        "  as the `all_available_square_rect_full_image_local_support_pairwise_source_trainonly` probe.",
        "- **QANet 20ep**: `qanet_score`, 20-epoch CNN on 64px crops with source identity.",
        "- **Hybrid α**: rank-based combination of `qanet_rank_in_cluster` and",
        "  `pairwise_cluster_rank`. α=0.00 = pure pairwise; α=1.00 = pure QANet.",
        "",
        "## Method Note",
        "",
        "- `fusion_only_*` and `full_pipeline_*` rows are NOT included here.",
        "- This table covers only QA filter diagnostics: hybrid alpha variants",
        "  and the two component signals (pairwise LR, QANet 20ep).",
        "- Do not label any alpha variant as 'our best' without cleaner protocol.",
        "",
        "## IoU Results",
        "",
        f"| Dataset | Fold | r | Gate | Pairwise LR | QANet 20ep | {alpha_headers} |",
        f"|---|---:|---:|:---:|---:|---:|{alpha_seps}|",
    ]

    for _, row in table.iterrows():
        fold_label = f"**{row['fold']}**" if row["fold"] == "Mean" else row["fold"]
        gate_sym = "✓" if row["gate_passed"] else "✗"
        alpha_vals = " | ".join(fmt(row[col]) for _, col in alpha_cols)
        lines.append(
            f"| {row['dataset']} | {fold_label}"
            f" | {fmtr(row['test_spearman_r'])}"
            f" | {gate_sym}"
            f" | {fmt(row['pairwise_lr_top3'])}"
            f" | {fmt(row['qanet_20ep_top3'])}"
            f" | {alpha_vals} |"
        )

    lines += [
        "",
        "## Delta vs Pairwise LR Baseline",
        "",
        f"| Dataset | Fold | {delta_headers} |",
        f"|---|---:|{delta_seps}|",
    ]
    for _, row in table.iterrows():
        fold_label = f"**{row['fold']}**" if row["fold"] == "Mean" else row["fold"]
        ref = row["pairwise_lr_top3"]
        delta_vals = " | ".join(fmtd(row[col], ref) for _, col in alpha_cols)
        lines.append(f"| {row['dataset']} | {fold_label} | {delta_vals} |")

    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- A hybrid score is only useful if it consistently beats pairwise LR",
        "  (α=0.00) across all datasets and both folds.",
        "- Gains on HSC/HeLa (where QANet 20ep is strong) may be offset by",
        "  losses on MSC (where QANet 20ep has a train/test appearance gap).",
        "- MSC fold-2 is below the gate (test r < 0.40) and should not be",
        "  included in a best-alpha conclusion.",
        "- If no alpha consistently beats α=0.00, the pairwise LR remains the",
        "  cleaner deployable score and QANet should be retrained.",
        "",
        "## Files",
        "",
        f"Rank correlation: `{HYBRID_PROBE_ROOT}/qanet_hybrid_rank_correlation.csv`",
        f"This comparison: `{HYBRID_PROBE_ROOT}/qanet_hybrid_comparison.csv`",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", metavar="DS")
    parser.add_argument("--folds", nargs="+", metavar="FOLD")
    parser.add_argument("--min-test-r", type=float, default=DEFAULT_MIN_TEST_R)
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-fusion", action="store_true")
    parser.add_argument("--force-hybrid-csv", action="store_true",
                        help="Recompute hybrid scored CSVs even if they exist.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    folds = FOLDS
    if args.datasets:
        folds = [f for f in folds if f["dataset"] in args.datasets]
    if args.folds:
        folds = [f for f in folds if f["fold"] in args.folds]

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Building hybrid scored CSVs")
    print("=" * 60)
    for entry in folds:
        label = f"{entry['dataset']}/{entry['fold']}"
        out_csv = hybrid_scored_csv(repo, entry)
        if out_csv.exists() and not args.force_hybrid_csv:
            print(f"  [DONE] {label} — hybrid CSV already exists, skipping.")
            continue
        build_hybrid_scored_csv(repo, entry)

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2: Rank correlations")
    print("=" * 60)
    corr_df = compute_rank_correlations(repo, folds)
    if corr_df.empty:
        print("[ERROR] No rank correlations available.")
        sys.exit(1)

    out_root = repo / HYBRID_PROBE_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    corr_csv = out_root / "qanet_hybrid_rank_correlation.csv"
    corr_df.to_csv(corr_csv, index=False)

    # Print test-split gate table
    test_corr = corr_df[corr_df["split"] == "test"].copy()
    for _, row in test_corr.sort_values(["dataset", "fold"]).iterrows():
        gate = "✓" if float(row["spearman_r"]) >= args.min_test_r else "✗ BELOW GATE"
        print(f"  {row['dataset']:22s} {row['fold']:6s} r={float(row['spearman_r']):.4f} {gate}")
    print(f"Wrote: {corr_csv}")

    # Build r_lookup
    r_lookup: dict[tuple, float] = {}
    for _, row in test_corr.iterrows():
        r_lookup[(row["dataset"], row["fold"])] = float(row["spearman_r"])

    if args.skip_fusion:
        print("\n[INFO] --skip-fusion set; stopping after rank correlations.")
        return

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 3: Fusion (gate: test Spearman r >= {args.min_test_r:.2f})")
    print("=" * 60)
    for entry in folds:
        label = f"{entry['dataset']}/{entry['fold']}"
        key = (entry["dataset"], entry["fold"])
        test_r = r_lookup.get(key, float("nan"))
        gate_passed = test_r >= args.min_test_r

        for alpha in HYBRID_ALPHAS:
            # alpha=0.0 (pairwise-only) runs for all folds as reference baseline
            if not gate_passed and alpha != 0.0:
                print(f"  [SKIP] {label} alpha={alpha:.2f} — below gate (r={test_r:.4f})")
                continue
            out_dir = hybrid_fusion_dir(repo, entry, alpha)
            comparison_csv = out_dir / "ranker_pruned_fusion_comparison.csv"
            if not args.force_fusion and comparison_csv.exists():
                print(f"  [DONE] {label} alpha={alpha:.2f} — fusion exists, skipping.")
                continue
            run_fusion_alpha(repo, entry, alpha, dry_run=args.dry_run)

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4: Building comparison table")
    print("=" * 60)
    table = build_comparison_table(repo, folds, corr_df, args.min_test_r)

    comparison_csv = out_root / "qanet_hybrid_comparison.csv"
    comparison_md = out_root / "qanet_hybrid_comparison.md"
    table.to_csv(comparison_csv, index=False)
    comparison_md.write_text(render_markdown(table, args.min_test_r))
    print(f"Wrote comparison CSV:      {comparison_csv}")
    print(f"Wrote comparison markdown: {comparison_md}")

    # Print summary
    print()
    alpha_cols = [f"hybrid_alpha_{a:.2f}".replace(".", "p") for a in HYBRID_ALPHAS]
    display_cols = ["dataset", "fold", "test_spearman_r", "pairwise_lr_top3", "qanet_20ep_top3"] + alpha_cols
    present_cols = [c for c in display_cols if c in table.columns]
    mean_rows = table[table["fold"] == "Mean"][present_cols]
    if not mean_rows.empty:
        print("Dataset-mean IoU (gated folds only for QANet-containing alphas):")
        print(mean_rows.to_string(index=False))
    print()

    # Determine best alpha (by mean over gated dataset means)
    mean_table = table[table["fold"] == "Mean"]
    best_alpha = None
    best_iou = float("-inf")
    for alpha in HYBRID_ALPHAS:
        col = f"hybrid_alpha_{alpha:.2f}".replace(".", "p")
        if col in mean_table.columns:
            mean_iou = mean_table[col].mean()
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_alpha = alpha

    pw_mean = mean_table["pairwise_lr_top3"].mean()
    print(f"Pairwise LR (α=0.00) overall dataset-mean IoU: {pw_mean:.6f}")
    print(f"Best hybrid alpha: α={best_alpha:.2f} with overall mean IoU: {best_iou:.6f}")
    delta = best_iou - pw_mean
    print(f"Delta best hybrid vs pairwise LR: {delta:+.6f}")
    if delta > 0.001:
        print("DIAGNOSTIC SIGNAL: Hybrid score may improve over pairwise LR.")
        print("  Verify per-dataset breakdown before promoting to paper row.")
    elif delta > 0.0:
        print("DIAGNOSTIC SIGNAL: Marginal hybrid improvement; within noise.")
    else:
        print("DIAGNOSTIC SIGNAL: Hybrid does not consistently improve over pairwise LR.")
        print("  QANet requires further training before hybrid is beneficial.")


if __name__ == "__main__":
    main()
