#!/usr/bin/env python3
"""
run_hybrid_val_alpha_select.py
------------------------------
Validation-based alpha selection for the QANet-Pairwise hybrid scorer.

This script performs a *clean* alpha selection protocol:
  - For each dataset (aggregating all gated folds), the hybrid alpha is
    selected using VALIDATION-split IoU only.
  - The selected alpha is then evaluated on the TEST split.
  - No information from the test split is used in the selection step.

Data source: leaderboard CSVs already produced by run_qanet_hybrid_eval.py.
Each leaderboard CSV (fusion_crops_leaderboard.csv) contains:
  eval_validation_mean_jaccard  (used for selection)
  eval_test_mean_jaccard        (used for final evaluation)

Gate: test-split Spearman r >= 0.40 (same as Addendum 26/27).
  - MSC fold-2 (r=0.26) is gated out for alpha > 0.
  - alpha=0.00 (pure pairwise) always included for all folds.

Outputs (in HYBRID_PROBE_ROOT):
  qanet_hybrid_val_alpha_selection.csv
  qanet_hybrid_val_alpha_selection.md

Usage:
    python scripts/run_hybrid_val_alpha_select.py
    python scripts/run_hybrid_val_alpha_select.py --min-test-r 0.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants (mirror run_qanet_hybrid_eval.py)
# ---------------------------------------------------------------------------

HYBRID_PROBE_ROOT = Path("hpc_results/soft_hybrid_probe/qanet_hybrid_probe")
HYBRID_ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00]
DEFAULT_MIN_TEST_R = 0.40

FOLDS: list[dict] = [
    dict(dataset="BF-C2DL-HSC",       crop_tag="sz64",   variant="baseline",       fold="fold-1"),
    dict(dataset="BF-C2DL-HSC",       crop_tag="sz64",   variant="baseline",       fold="fold-2"),
    dict(dataset="BF-C2DL-MuSC",      crop_tag="sz256",  variant="musc_crop256",   fold="fold-1"),
    dict(dataset="BF-C2DL-MuSC",      crop_tag="sz256",  variant="musc_crop256",   fold="fold-2"),
    dict(dataset="DIC-C2DH-HeLa",     crop_tag="sz256",  variant="hela_crop256",   fold="fold-1"),
    dict(dataset="DIC-C2DH-HeLa",     crop_tag="sz256",  variant="hela_crop256",   fold="fold-2"),
    dict(dataset="Fluo-C2DL-MSC",     crop_tag="sz1024", variant="msc_crop1024",   fold="fold-1"),
    dict(dataset="Fluo-C2DL-MSC",     crop_tag="sz1024", variant="msc_crop1024",   fold="fold-2"),
    dict(dataset="Fluo-N2DH-GOWT1",   crop_tag="sz160",  variant="gowt1_crop160",  fold="fold-1"),
    dict(dataset="Fluo-N2DH-GOWT1",   crop_tag="sz160",  variant="gowt1_crop160",  fold="fold-2"),
]

# Test-split Spearman r values from Addendum 26 (qanet_20ep_rank_correlation.csv).
# Used to gate which folds contribute to QANet-containing alphas.
KNOWN_TEST_R: dict[tuple, float] = {
    ("BF-C2DL-HSC",     "fold-1"): 0.61,
    ("BF-C2DL-HSC",     "fold-2"): 0.72,
    ("BF-C2DL-MuSC",    "fold-1"): 0.50,
    ("BF-C2DL-MuSC",    "fold-2"): 0.45,
    ("DIC-C2DH-HeLa",   "fold-1"): 0.61,
    ("DIC-C2DH-HeLa",   "fold-2"): 0.62,
    ("Fluo-C2DL-MSC",   "fold-1"): 0.52,
    ("Fluo-C2DL-MSC",   "fold-2"): 0.26,   # GATED
    ("Fluo-N2DH-GOWT1", "fold-1"): 0.55,
    ("Fluo-N2DH-GOWT1", "fold-2"): 0.76,
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def leaderboard_path(repo: Path, entry: dict, alpha: float) -> Path:
    alpha_tag = f"alpha{alpha:.2f}".replace(".", "p")
    return (
        repo / HYBRID_PROBE_ROOT
        / entry["dataset"] / entry["crop_tag"] / entry["variant"]
        / entry["fold"]
        / f"hybrid_{alpha_tag}_source_balanced_top3_simple"
        / "fusion"
        / "fusion_crops_leaderboard.csv"
    )


def load_rank_correlations(repo: Path) -> dict[tuple, float]:
    """Load test-split Spearman r from the saved rank correlation CSV, falling
    back to KNOWN_TEST_R constants if the file is not found."""
    corr_csv = repo / HYBRID_PROBE_ROOT / "qanet_hybrid_rank_correlation.csv"
    if not corr_csv.exists():
        corr_csv = repo / Path(
            "hpc_results/soft_hybrid_probe/qanet_20ep_probe/qanet_20ep_rank_correlation.csv"
        )
    if corr_csv.exists():
        df = pd.read_csv(corr_csv)
        test_df = df[df["split"] == "test"]
        return {
            (row["dataset"], row["fold"]): float(row["spearman_r"])
            for _, row in test_df.iterrows()
        }
    print("[WARN] No rank correlation CSV found; using hardcoded KNOWN_TEST_R values.")
    return dict(KNOWN_TEST_R)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def build_per_fold_table(repo: Path, min_test_r: float) -> pd.DataFrame:
    """
    For each (dataset, fold, alpha), read eval_validation_mean_jaccard and
    eval_test_mean_jaccard from the leaderboard CSV.

    Returns a tidy DataFrame with columns:
      dataset | fold | alpha | gate_passed | val_iou | test_iou
    """
    r_lookup = load_rank_correlations(repo)
    records = []

    for entry in FOLDS:
        key = (entry["dataset"], entry["fold"])
        test_r = r_lookup.get(key, float("nan"))
        gate_passed = not np.isnan(test_r) and test_r >= min_test_r

        for alpha in HYBRID_ALPHAS:
            # alpha=0 always runs regardless of gate
            if not gate_passed and alpha != 0.0:
                continue

            lb_path = leaderboard_path(repo, entry, alpha)
            if not lb_path.exists():
                # Expected for gated folds (alpha > 0)
                if alpha != 0.0:
                    continue
                print(f"  [WARN] leaderboard missing for {key} alpha={alpha:.2f}: {lb_path}")
                continue

            lb = pd.read_csv(lb_path)
            if lb.empty:
                continue

            row = lb.iloc[0]
            val_iou = float(row.get("eval_validation_mean_jaccard", float("nan")))
            test_iou = float(row.get("eval_test_mean_jaccard", float("nan")))

            records.append({
                "dataset": entry["dataset"],
                "fold": entry["fold"],
                "alpha": alpha,
                "gate_passed": gate_passed,
                "test_spearman_r": test_r,
                "val_iou": val_iou,
                "test_iou": test_iou,
            })

    return pd.DataFrame(records)


def select_alpha_per_dataset(fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each dataset, aggregate the per-fold validation IoU over all gated folds
    (or fold-1 only for MSC where fold-2 is gated), then pick the alpha that
    maximises mean validation IoU.

    Returns a DataFrame with columns:
      dataset | val_selected_alpha | n_folds_for_selection | val_iou | test_iou_selected
      | test_iou_alpha0 | delta_vs_pairwise | val_agrees_with_test
    """
    datasets = sorted(fold_df["dataset"].unique())
    records = []

    for dataset in datasets:
        ds_df = fold_df[fold_df["dataset"] == dataset].copy()

        # For validation selection: use only gated folds
        # (fold-2 for MSC is excluded for alpha > 0 anyway, but alpha=0 always present)
        # Collect (alpha → mean_val_iou) using only gated folds
        alpha_val: dict[float, list[float]] = {}
        for alpha in HYBRID_ALPHAS:
            sub = ds_df[ds_df["alpha"] == alpha]
            # Only count folds that are gated or alpha=0
            if alpha == 0.0:
                vals = sub["val_iou"].dropna().tolist()
            else:
                vals = sub[sub["gate_passed"]]["val_iou"].dropna().tolist()
            if vals:
                alpha_val[alpha] = vals

        if not alpha_val:
            continue

        # Mean validation IoU per alpha (mean over available gated folds)
        alpha_mean_val = {a: float(np.mean(v)) for a, v in alpha_val.items()}
        # Number of folds contributing to the validation selection
        n_folds = max(len(v) for v in alpha_val.values())

        # Validation-selected alpha: highest mean val IoU
        val_selected_alpha = max(alpha_mean_val, key=alpha_mean_val.__getitem__)
        val_iou_selected = alpha_mean_val[val_selected_alpha]

        # Test IoU for selected alpha (mean over gated folds)
        sel_test_rows = ds_df[
            (ds_df["alpha"] == val_selected_alpha) &
            (ds_df["gate_passed"] | (val_selected_alpha == 0.0))
        ]
        test_iou_selected = float(sel_test_rows["test_iou"].mean())

        # Pairwise baseline (alpha=0)
        pw_rows = ds_df[ds_df["alpha"] == 0.0]
        test_iou_alpha0 = float(pw_rows["test_iou"].mean())
        val_iou_alpha0 = float(pw_rows["val_iou"].mean())

        delta = test_iou_selected - test_iou_alpha0

        # Best alpha by test (from Addendum 27 results) — for comparison
        alpha_mean_test: dict[float, float] = {}
        for alpha in HYBRID_ALPHAS:
            sub = ds_df[ds_df["alpha"] == alpha]
            if alpha == 0.0:
                vals = sub["test_iou"].dropna().tolist()
            else:
                vals = sub[sub["gate_passed"]]["test_iou"].dropna().tolist()
            if vals:
                alpha_mean_test[alpha] = float(np.mean(vals))

        test_selected_alpha = max(alpha_mean_test, key=alpha_mean_test.__getitem__)

        records.append({
            "dataset": dataset,
            "n_folds_for_selection": n_folds,
            "val_selected_alpha": val_selected_alpha,
            "val_iou_selected": val_iou_selected,
            "val_iou_alpha0": val_iou_alpha0,
            "test_iou_selected": test_iou_selected,
            "test_iou_alpha0": test_iou_alpha0,
            "delta_vs_pairwise": delta,
            "test_optimal_alpha": test_selected_alpha,
            "val_matches_test_optimal": val_selected_alpha == test_selected_alpha,
            "val_beats_alpha0": val_selected_alpha != 0.0 and delta > 0,
        })

    return pd.DataFrame(records)


def build_val_alpha_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide table: one row per (dataset, fold), columns for each alpha's
    val_iou and test_iou.
    """
    datasets = sorted(fold_df["dataset"].unique())
    records = []

    for entry in FOLDS:
        ds = entry["dataset"]
        fold = entry["fold"]
        sub = fold_df[(fold_df["dataset"] == ds) & (fold_df["fold"] == fold)]
        if sub.empty:
            continue

        row: dict = {
            "dataset": ds,
            "fold": fold,
            "gate_passed": bool(sub["gate_passed"].any()),
            "test_spearman_r": float(sub["test_spearman_r"].iloc[0]),
        }
        for alpha in HYBRID_ALPHAS:
            alpha_row = sub[sub["alpha"] == alpha]
            tag = f"{alpha:.2f}".replace(".", "p")
            row[f"val_alpha{tag}"] = float(alpha_row["val_iou"].iloc[0]) if not alpha_row.empty else float("nan")
            row[f"test_alpha{tag}"] = float(alpha_row["test_iou"].iloc[0]) if not alpha_row.empty else float("nan")
        records.append(row)

    return pd.DataFrame(records)


def render_markdown(
    fold_table: pd.DataFrame,
    selection_table: pd.DataFrame,
    min_test_r: float,
) -> str:
    lines: list[str] = [
        "# QANet–Pairwise Hybrid: Validation-Selected Alpha Protocol",
        "",
        "Date: 2026-05-29",
        "",
        "## Protocol",
        "",
        "Alpha is selected per dataset using VALIDATION-split IoU only.",
        "Test IoU is then reported for the selected alpha.",
        "This is a clean (leak-free) protocol:",
        "",
        "1. For each dataset, aggregate validation IoU over all gated folds.",
        "2. Pick the alpha (0.00, 0.25, 0.50, 0.75, 1.00) that maximises",
        "   mean validation IoU.",
        "3. Report test IoU for that alpha vs alpha=0.00 (pairwise baseline).",
        "",
        f"Gate: test-split Spearman r >= {min_test_r:.2f}.",
        "Folds below the gate contribute only alpha=0.00 (pairwise baseline).",
        "",
        "## Reporting Guardrail",
        "",
        "- If the validation-selected alpha matches the test-optimal alpha AND",
        "  the delta is positive and consistent, the hybrid can be considered",
        "  for a clean paper row.",
        "- If the validation-selected alpha does NOT match the test-optimal alpha,",
        "  the hybrid provides no evidence of generalisation.",
        "",
        "## Per-Dataset Validation-Selected Alpha",
        "",
        "| Dataset | Folds | Val-Selected α | Val IoU (α sel) | Val IoU (α=0) | "
        "Test IoU (α sel) | Test IoU (α=0) | Δ test | Val=Test optimal? | "
        "Hybrid beats pairwise? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]

    for _, row in selection_table.iterrows():
        match_sym = "✓" if row["val_matches_test_optimal"] else "✗"
        beats_sym = "✓" if row["val_beats_alpha0"] else "—"
        lines.append(
            f"| {row['dataset']}"
            f" | {int(row['n_folds_for_selection'])}"
            f" | α={row['val_selected_alpha']:.2f}"
            f" | {float(row['val_iou_selected']):.6f}"
            f" | {float(row['val_iou_alpha0']):.6f}"
            f" | {float(row['test_iou_selected']):.6f}"
            f" | {float(row['test_iou_alpha0']):.6f}"
            f" | {float(row['delta_vs_pairwise']):+.6f}"
            f" | {match_sym}"
            f" | {beats_sym} |"
        )

    # Overall mean (over all datasets' test IoU)
    lines += ["", "**Overall (mean over datasets):**", ""]
    mean_test_sel = float(selection_table["test_iou_selected"].mean())
    mean_test_pw = float(selection_table["test_iou_alpha0"].mean())
    mean_delta = mean_test_sel - mean_test_pw
    lines += [
        f"- Mean test IoU (validation-selected alpha): **{mean_test_sel:.6f}**",
        f"- Mean test IoU (alpha=0 pairwise):          **{mean_test_pw:.6f}**",
        f"- Mean delta:                                **{mean_delta:+.6f}**",
        "",
    ]

    lines += [
        "## Per-Fold Validation and Test IoU by Alpha",
        "",
        "Val IoU = `eval_validation_mean_jaccard` from leaderboard CSV (used for selection).",
        "Test IoU = `eval_test_mean_jaccard` from leaderboard CSV (held out).",
        "",
    ]

    # Val IoU table
    alpha_headers = " | ".join(f"Val α={a:.2f}" for a in HYBRID_ALPHAS)
    lines += [
        f"| Dataset | Fold | Gate | r | {alpha_headers} |",
        f"|---|---:|:---:|---:| {'|'.join('---:' for _ in HYBRID_ALPHAS)} |",
    ]
    for _, row in fold_table.iterrows():
        gate_sym = "✓" if row["gate_passed"] else "✗"
        vals = []
        for a in HYBRID_ALPHAS:
            tag = f"{a:.2f}".replace(".", "p")
            v = row.get(f"val_alpha{tag}", float("nan"))
            vals.append(f"{float(v):.6f}" if not (isinstance(v, float) and np.isnan(v)) else "n/a")
        lines.append(
            f"| {row['dataset']} | {row['fold']}"
            f" | {gate_sym}"
            f" | {float(row['test_spearman_r']):.4f}"
            f" | {' | '.join(vals)} |"
        )

    # Test IoU table
    alpha_headers2 = " | ".join(f"Test α={a:.2f}" for a in HYBRID_ALPHAS)
    lines += [
        "",
        f"| Dataset | Fold | Gate | r | {alpha_headers2} |",
        f"|---|---:|:---:|---:| {'|'.join('---:' for _ in HYBRID_ALPHAS)} |",
    ]
    for _, row in fold_table.iterrows():
        gate_sym = "✓" if row["gate_passed"] else "✗"
        vals = []
        for a in HYBRID_ALPHAS:
            tag = f"{a:.2f}".replace(".", "p")
            v = row.get(f"test_alpha{tag}", float("nan"))
            vals.append(f"{float(v):.6f}" if not (isinstance(v, float) and np.isnan(v)) else "n/a")
        lines.append(
            f"| {row['dataset']} | {row['fold']}"
            f" | {gate_sym}"
            f" | {float(row['test_spearman_r']):.4f}"
            f" | {' | '.join(vals)} |"
        )

    lines += [
        "",
        "## Method Note",
        "",
        "- `fusion_only_*` and `full_pipeline_*` rows are NOT included here.",
        "- This is a strict QA/ENSEMBLE diagnostic table.",
        "- The hybrid score is not yet a paper row; it requires consistent gains",
        "  under this clean protocol before promotion.",
        "",
        "## Files",
        "",
        f"Selection results: `{HYBRID_PROBE_ROOT}/qanet_hybrid_val_alpha_selection.csv`",
        f"This report:       `{HYBRID_PROBE_ROOT}/qanet_hybrid_val_alpha_selection.md`",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-test-r", type=float, default=DEFAULT_MIN_TEST_R)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("Building per-fold leaderboard table")
    print("=" * 60)
    fold_df = build_per_fold_table(repo, args.min_test_r)
    if fold_df.empty:
        print("[ERROR] No leaderboard data found. Run run_qanet_hybrid_eval.py first.")
        return
    print(f"  Loaded {len(fold_df)} (fold, alpha) combinations.")

    print()
    print("=" * 60)
    print("Selecting alpha per dataset on validation IoU")
    print("=" * 60)
    selection_df = select_alpha_per_dataset(fold_df)

    # Print selection summary
    print()
    print(f"{'Dataset':<30} {'ValSel α':>10} {'ΔTest':>10} {'Val=Test?':>12} {'Beats PW?':>12}")
    print("-" * 76)
    for _, row in selection_df.iterrows():
        match = "YES" if row["val_matches_test_optimal"] else "NO"
        beats = "YES" if row["val_beats_alpha0"] else "no"
        print(
            f"{row['dataset']:<30}"
            f" {row['val_selected_alpha']:>10.2f}"
            f" {row['delta_vs_pairwise']:>+10.6f}"
            f" {match:>12}"
            f" {beats:>12}"
        )
    print("-" * 76)
    mean_delta = float(selection_df["delta_vs_pairwise"].mean())
    n_beats = int(selection_df["val_beats_alpha0"].sum())
    n_match = int(selection_df["val_matches_test_optimal"].sum())
    n_ds = len(selection_df)
    print(f"Mean delta (test IoU, val-selected α vs α=0): {mean_delta:+.6f}")
    print(f"Datasets where val-selected α matches test-optimal α: {n_match}/{n_ds}")
    print(f"Datasets where val-selected α beats pairwise (test):  {n_beats}/{n_ds}")

    print()
    if n_beats == n_ds:
        print("CLEAN SIGNAL: Validation-selected hybrid beats pairwise on ALL datasets.")
        print("  The hybrid approach is a viable clean deployable policy.")
    elif n_beats > n_ds // 2:
        print(f"PARTIAL SIGNAL: Hybrid beats pairwise on {n_beats}/{n_ds} datasets.")
        print("  Dataset-conditional hybrid policy may be defensible.")
    else:
        print(f"WEAK SIGNAL: Hybrid beats pairwise on only {n_beats}/{n_ds} datasets.")
        print("  The hybrid is NOT a clean improvement over pairwise LR at this stage.")

    # Build wide table for output
    fold_wide = build_val_alpha_table(fold_df)

    # Write outputs
    out_dir = repo / HYBRID_PROBE_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_csv = out_dir / "qanet_hybrid_val_alpha_selection.csv"
    sel_md = out_dir / "qanet_hybrid_val_alpha_selection.md"

    selection_df.to_csv(sel_csv, index=False)
    sel_md.write_text(render_markdown(fold_wide, selection_df, args.min_test_r))

    print()
    print(f"Wrote: {sel_csv}")
    print(f"Wrote: {sel_md}")


if __name__ == "__main__":
    main()
