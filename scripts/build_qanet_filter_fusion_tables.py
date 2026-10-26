#!/usr/bin/env python3
"""Build Step 3 comparison tables for QANet-filter vs tabular rankers.

This script collects SIMPLE fusion results from:
  - QANet-filtered candidate parquets (score_column=qanet_score)
  - MLP tabular ranker candidates (score_column=candidate_ranker_score)

and compares them against the fixed pairwise top3 (LogisticRegression) and
source-prior top5 baselines on the same dataset/fold combinations.

The comparison covers MuSC and MSC only (the two datasets used in the QANet
smoke-pass and any follow-up QANet probes).

Output:
    hpc_results/soft_hybrid_probe/qanet_filter_probe/
        qanet_vs_ranker_comparison.csv
        qanet_vs_ranker_comparison.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROBE_ROOT = Path(
    "hpc_results/soft_hybrid_probe/qanet_filter_probe"
)
BASELINE_COMPARISON_CSV = Path(
    "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
    "all_available_simple_policy_comparison_rows.csv"
)
SOURCE_PRIOR_COMPARISON_CSV = Path(
    "hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/"
    "source_prior_topk5_simple/source_prior_topk5_simple_comparison_rows.csv"
)

# Dataset/fold/path tuples for QANet smoke-pass runs.
QANET_FOLD_SPECS: list[dict] = [
    {
        "dataset": "BF-C2DL-MuSC",
        "crop_tag": "sz256",
        "variant": "musc_crop256",
        "fold": "fold-1",
        "comparison_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "BF-C2DL-MuSC/sz256/musc_crop256/fold-1/"
            "qanet_source_balanced_top3_simple/ranker_pruned_fusion_comparison.csv"
        ),
    },
    {
        "dataset": "BF-C2DL-MuSC",
        "crop_tag": "sz256",
        "variant": "musc_crop256",
        "fold": "fold-2",
        "comparison_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "BF-C2DL-MuSC/sz256/musc_crop256/fold-2/"
            "qanet_source_balanced_top3_simple/ranker_pruned_fusion_comparison.csv"
        ),
    },
    {
        "dataset": "Fluo-C2DL-MSC",
        "crop_tag": "sz1024",
        "variant": "msc_crop1024",
        "fold": "fold-1",
        "comparison_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-1/"
            "qanet_source_balanced_top3_simple/ranker_pruned_fusion_comparison.csv"
        ),
    },
    {
        "dataset": "Fluo-C2DL-MSC",
        "crop_tag": "sz1024",
        "variant": "msc_crop1024",
        "fold": "fold-2",
        "comparison_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-2/"
            "qanet_source_balanced_top3_simple/ranker_pruned_fusion_comparison.csv"
        ),
    },
]

# Dataset/fold/path tuples for MLP (Ridge/MLP regression) ranker runs.
MLP_FOLD_SPECS: list[dict] = [
    {
        "dataset": "BF-C2DL-MuSC",
        "crop_tag": "sz256",
        "variant": "musc_crop256",
        "fold": "fold-1",
        "comparison_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "BF-C2DL-MuSC/sz256/musc_crop256/fold-1/"
            "mlp_source_balanced_top3_simple/ranker_pruned_fusion_comparison.csv"
        ),
    },
    {
        "dataset": "BF-C2DL-MuSC",
        "crop_tag": "sz256",
        "variant": "musc_crop256",
        "fold": "fold-2",
        "comparison_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "BF-C2DL-MuSC/sz256/musc_crop256/fold-2/"
            "mlp_source_balanced_top3_simple/ranker_pruned_fusion_comparison.csv"
        ),
    },
    {
        "dataset": "Fluo-C2DL-MSC",
        "crop_tag": "sz1024",
        "variant": "msc_crop1024",
        "fold": "fold-1",
        "comparison_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-1/"
            "mlp_source_balanced_top3_simple/ranker_pruned_fusion_comparison.csv"
        ),
    },
    {
        "dataset": "Fluo-C2DL-MSC",
        "crop_tag": "sz1024",
        "variant": "msc_crop1024",
        "fold": "fold-2",
        "comparison_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-2/"
            "mlp_source_balanced_top3_simple/ranker_pruned_fusion_comparison.csv"
        ),
    },
]


def load_qanet_rows(fold_specs: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in fold_specs:
        p = Path(spec["comparison_csv"])
        if not p.exists():
            print(f"  WARNING: missing QANet comparison CSV: {p}")
            continue
        df = pd.read_csv(p)
        df = df.assign(
            qanet_probe="smoke_2epoch_64px",
        )
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No QANet comparison CSVs found.")
    return pd.concat(rows, ignore_index=True)


def load_mlp_rows(fold_specs: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in fold_specs:
        p = Path(spec["comparison_csv"])
        if not p.exists():
            print(f"  WARNING: missing MLP comparison CSV: {p}")
            continue
        df = pd.read_csv(p)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_pairwise_baseline(datasets: list[str]) -> pd.DataFrame:
    if not BASELINE_COMPARISON_CSV.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {BASELINE_COMPARISON_CSV}")
    df = pd.read_csv(BASELINE_COMPARISON_CSV)
    mask = df["dataset"].isin(datasets) & (df["policy"] == "source_balanced_topk")
    return df[mask].copy()


def load_source_prior_baseline(datasets: list[str]) -> pd.DataFrame:
    if not SOURCE_PRIOR_COMPARISON_CSV.exists():
        print(f"  WARNING: source-prior CSV not found: {SOURCE_PRIOR_COMPARISON_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(SOURCE_PRIOR_COMPARISON_CSV)
    mask = df["dataset"].isin(datasets)
    return df[mask].copy()


def build_comparison_table(
    qanet_rows: pd.DataFrame,
    pairwise_rows: pd.DataFrame,
    source_prior_rows: pd.DataFrame,
    mlp_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build fold-level comparison table."""
    key_cols = ["dataset", "fold"]

    pairwise_lookup = pairwise_rows.set_index(key_cols)["mean_iou"].to_dict()
    sp_lookup = (
        source_prior_rows.set_index(key_cols)["mean_iou"].to_dict()
        if not source_prior_rows.empty
        else {}
    )
    mlp_lookup = (
        mlp_rows.set_index(key_cols)["mean_iou"].to_dict()
        if not mlp_rows.empty
        else {}
    )

    records = []
    for _, row in qanet_rows.iterrows():
        key = (row["dataset"], row["fold"])
        pairwise_iou = pairwise_lookup.get(key, float("nan"))
        sp_iou = sp_lookup.get(key, float("nan"))
        mlp_iou = mlp_lookup.get(key, float("nan"))
        qanet_iou = float(row["mean_iou"])
        records.append(
            {
                "dataset": row["dataset"],
                "fold": row["fold"],
                "pairwise_lr_top3_iou": pairwise_iou,
                "mlp_tabular_top3_iou": mlp_iou,
                "source_prior_top5_iou": sp_iou,
                "qanet_smoke_top3_iou": qanet_iou,
                "delta_qanet_vs_pairwise": qanet_iou - pairwise_iou,
                "delta_qanet_vs_mlp": qanet_iou - mlp_iou,
                "delta_mlp_vs_pairwise": mlp_iou - pairwise_iou,
                "qanet_method_key": row.get("method_key", ""),
                "pairwise_rows": int(row.get("filtered_candidate_rows", 0)),
                "input_rows": int(row.get("input_candidate_rows", 0)),
            }
        )

    # Add dataset mean rows.
    fold_df = pd.DataFrame(records)
    mean_rows = []
    for dataset, grp in fold_df.groupby("dataset"):
        mean_rows.append(
            {
                "dataset": dataset,
                "fold": "Mean",
                "pairwise_lr_top3_iou": grp["pairwise_lr_top3_iou"].mean(),
                "mlp_tabular_top3_iou": grp["mlp_tabular_top3_iou"].mean(),
                "source_prior_top5_iou": grp["source_prior_top5_iou"].mean(),
                "qanet_smoke_top3_iou": grp["qanet_smoke_top3_iou"].mean(),
                "delta_qanet_vs_pairwise": grp["delta_qanet_vs_pairwise"].mean(),
                "delta_qanet_vs_mlp": grp["delta_qanet_vs_mlp"].mean(),
                "delta_mlp_vs_pairwise": grp["delta_mlp_vs_pairwise"].mean(),
                "qanet_method_key": "",
                "pairwise_rows": 0,
                "input_rows": 0,
            }
        )
    all_rows = pd.concat(
        [fold_df, pd.DataFrame(mean_rows)], ignore_index=True
    )

    # Sort: folds before means, datasets alphabetically.
    all_rows["_sort_dataset"] = all_rows["dataset"]
    all_rows["_sort_fold"] = all_rows["fold"].apply(
        lambda x: "z" + x if x == "Mean" else x
    )
    all_rows = all_rows.sort_values(
        ["_sort_dataset", "_sort_fold"]
    ).drop(columns=["_sort_dataset", "_sort_fold"])
    return all_rows.reset_index(drop=True)


def render_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# QANet Smoke-Pass vs Tabular Rankers: Step 3 Comparison",
        "",
        "Date: 2026-05-28",
        "",
        "## Scope",
        "",
        "This table compares the QANet smoke-pass (2 epochs, 64 px crops, CPU)",
        "against three tabular/prior baselines on MuSC and MSC folds.",
        "These are internal-diagnostic rows only; the smoke-pass QANet is NOT",
        "a paper-ready result.",
        "",
        "- **Pairwise LR top3**: score_column=pairwise_cluster_score",
        "  (LogisticRegression on tabular candidate features within clusters),",
        "  top_k=3, source_top_k=1, SIMPLE fusion.",
        "- **MLP tabular top3**: score_column=candidate_ranker_score",
        "  (MLP/Ridge regression on tabular candidate features),",
        "  top_k=3, source_top_k=1, SIMPLE fusion.",
        "- **Source-prior top5**: score_column=source_train_mean_iou, top_k=5,",
        "  source_top_k=1, SIMPLE fusion.",
        "- **QANet smoke top3**: score_column=qanet_score (2-epoch 64px CNN),",
        "  top_k=3, source_top_k=1, SIMPLE fusion.",
        "",
        "## Results",
        "",
        "| Dataset | Fold | Pairwise LR top3 | MLP tabular top3 | Source-prior top5 | QANet smoke top3 | Δ (QANet vs LR) | Δ (QANet vs MLP) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        dataset = row["dataset"]
        fold = row["fold"]

        def fmt(val: object) -> str:
            return f"{val:.6f}" if not pd.isna(val) else "n/a"

        def fmtd(val: object) -> str:
            return f"{val:+.6f}" if not pd.isna(val) else "n/a"

        fold_label = f"**{fold}**" if fold == "Mean" else fold
        lines.append(
            f"| {dataset} | {fold_label}"
            f" | {fmt(row['pairwise_lr_top3_iou'])}"
            f" | {fmt(row['mlp_tabular_top3_iou'])}"
            f" | {fmt(row['source_prior_top5_iou'])}"
            f" | {fmt(row['qanet_smoke_top3_iou'])}"
            f" | {fmtd(row['delta_qanet_vs_pairwise'])}"
            f" | {fmtd(row['delta_qanet_vs_mlp'])}"
            f" |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- The smoke-pass QANet (2 CPU epochs, 64 px) does not consistently",
        "  improve over either tabular ranker. Results are neutral for MuSC",
        "  and regressive for MSC (especially fold-2).",
        "- **MLP vs Pairwise LR**: Both tabular methods use the same engineered",
        "  features; pairwise LR trains a within-cluster preference model while",
        "  MLP trains a pointwise IoU regressor.",
        "- **QANet direction is not invalidated.** The smoke model is severely",
        "  undertrained and uses a small crop size. The rank correlation between",
        "  the QANet scores and true IoU should be inspected (see",
        "  qanet_rank_correlation.csv) before drawing conclusions.",
        "- Next step: train a QANet with >= 20 epochs, 128-256 px crops, and",
        "  verify Spearman r >= 0.40 on test split before running fusion.",
        "",
        "## Files",
        "",
        "QANet smoke comparison CSVs:",
        "",
    ]
    for spec in QANET_FOLD_SPECS:
        lines.append(f"- `{spec['comparison_csv']}`")

    lines += [
        "",
        "MLP tabular ranker comparison CSVs:",
        "",
    ]
    for spec in MLP_FOLD_SPECS:
        lines.append(f"- `{spec['comparison_csv']}`")

    lines += [
        "",
        "Pairwise LR baseline comparison CSV:",
        f"- `{BASELINE_COMPARISON_CSV}`",
        "",
        "Source-prior baseline CSV:",
        f"- `{SOURCE_PRIOR_COMPARISON_CSV}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROBE_ROOT,
        help="Output directory for comparison CSVs and markdown.",
    )
    args = parser.parse_args()

    datasets = sorted({spec["dataset"] for spec in QANET_FOLD_SPECS})

    print("Loading QANet comparison rows...")
    qanet_rows = load_qanet_rows(QANET_FOLD_SPECS)
    print(f"  {len(qanet_rows)} QANet fold rows loaded.")

    print("Loading MLP tabular ranker rows...")
    mlp_rows = load_mlp_rows(MLP_FOLD_SPECS)
    print(f"  {len(mlp_rows)} MLP ranker fold rows loaded.")

    print("Loading pairwise LR baseline...")
    pairwise_rows = load_pairwise_baseline(datasets)
    print(f"  {len(pairwise_rows)} pairwise LR baseline rows loaded.")

    print("Loading source-prior baseline...")
    sp_rows = load_source_prior_baseline(datasets)
    print(f"  {len(sp_rows)} source-prior rows loaded.")

    print("Building comparison table...")
    table = build_comparison_table(qanet_rows, pairwise_rows, sp_rows, mlp_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "qanet_vs_ranker_comparison.csv"
    md_path = args.output_dir / "qanet_vs_ranker_comparison.md"

    table.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(table))

    print(f"\nWrote comparison CSV: {csv_path}")
    print(f"Wrote comparison markdown: {md_path}")
    print()

    # Print the fold + mean rows
    display_cols = [
        "dataset",
        "fold",
        "pairwise_lr_top3_iou",
        "mlp_tabular_top3_iou",
        "source_prior_top5_iou",
        "qanet_smoke_top3_iou",
        "delta_qanet_vs_pairwise",
        "delta_qanet_vs_mlp",
    ]
    print(table[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
