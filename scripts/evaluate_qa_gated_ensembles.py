"""
Script to evaluate QA-gated ensembles and individual competitors.
Computes mean Jaccard (IoU) and F1 scores across all splits.
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import data_processing.utils.dataset_dataframe_creation as ddc


def evaluate_qa_ensembles(qa_folder: Path) -> pd.DataFrame:
    """
    Evaluate all QA-gated ensemble parquet files.

    Returns a DataFrame with mean scores per ensemble config and split.
    """
    results = []

    parquet_files = sorted(qa_folder.glob("*.parquet"))

    for parquet_path in parquet_files:
        df = ddc.load_dataframe_from_parquet_with_metadata(str(parquet_path))

        # Extract config name from filename
        config_name = parquet_path.stem

        # Find the IoU/F1 columns (format: M1--_iou, M2--_iou, etc.)
        iou_cols = [c for c in df.columns if c.endswith("_iou")]
        f1_cols = [c for c in df.columns if c.endswith("_f1")]

        if not iou_cols:
            print(f"Warning: No IoU columns found in {parquet_path}")
            continue

        # Use the first IoU/F1 column (assuming one metric per file)
        iou_col = iou_cols[0]
        f1_col = f1_cols[0] if f1_cols else None

        # Get unique splits
        splits = df["split"].unique()

        for split in splits:
            split_df = df[df["split"] == split]

            result = {
                "config": config_name,
                "split": split,
                "iou_mean": split_df[iou_col].mean(),
                "iou_std": split_df[iou_col].std(),
                "count": len(split_df),
            }

            if f1_col:
                result["f1_mean"] = split_df[f1_col].mean()
                result["f1_std"] = split_df[f1_col].std()

            results.append(result)

        # Also compute overall (all splits combined)
        result = {
            "config": config_name,
            "split": "all",
            "iou_mean": df[iou_col].mean(),
            "iou_std": df[iou_col].std(),
            "count": len(df),
        }
        if f1_col:
            result["f1_mean"] = df[f1_col].mean()
            result["f1_std"] = df[f1_col].std()
        results.append(result)

    return pd.DataFrame(results)


def evaluate_individual_competitors(fused_split_path: Path) -> pd.DataFrame:
    """
    Evaluate individual competitors from the fused_split parquet.

    Returns a DataFrame with mean scores per competitor and split.
    """
    results = []

    df = ddc.load_dataframe_from_parquet_with_metadata(str(fused_split_path))

    competitors = df["competitor"].unique()
    splits = df["split"].unique()

    for competitor in competitors:
        competitor_df = df[df["competitor"] == competitor]

        for split in splits:
            split_df = competitor_df[competitor_df["split"] == split]

            if len(split_df) == 0:
                continue

            result = {
                "competitor": competitor,
                "split": split,
                "jaccard_mean": split_df["jaccard_score"].mean(),
                "jaccard_std": split_df["jaccard_score"].std(),
                "f1_mean": split_df["f1_score"].mean(),
                "f1_std": split_df["f1_score"].std(),
                "count": len(split_df),
            }
            results.append(result)

        # Also compute overall (all splits combined)
        result = {
            "competitor": competitor,
            "split": "all",
            "jaccard_mean": competitor_df["jaccard_score"].mean(),
            "jaccard_std": competitor_df["jaccard_score"].std(),
            "f1_mean": competitor_df["f1_score"].mean(),
            "f1_std": competitor_df["f1_score"].std(),
            "count": len(competitor_df),
        }
        results.append(result)

    return pd.DataFrame(results)


def format_results_table(
    df: pd.DataFrame, value_cols: list[str], group_col: str
) -> None:
    """Print a formatted results table."""
    splits_order = ["train", "validation", "test", "all"]

    for split in splits_order:
        split_df = df[df["split"] == split].copy()
        if len(split_df) == 0:
            continue

        print(f"\n{'='*60}")
        print(f"Split: {split.upper()}")
        print("=" * 60)

        # Sort by first value column (descending)
        split_df = split_df.sort_values(value_cols[0], ascending=False)

        # Print header
        header = f"{group_col:40s}"
        for col in value_cols:
            header += f" | {col:12s}"
        header += f" | {'count':>6s}"
        print(header)
        print("-" * len(header))

        # Print rows
        for _, row in split_df.iterrows():
            line = f"{row[group_col]:40s}"
            for col in value_cols:
                line += f" | {row[col]:12.4f}"
            line += f" | {row['count']:>6d}"
            print(line)


def main():
    project_root = Path(__file__).parent.parent

    qa_folder = project_root / "C1_ds1-42-7015_QA"
    qa_folder = project_root / "C1_ds1-42-7015_QA-eb7-1"
    fused_split_path = project_root / "fused_split.parquet"

    print("=" * 80)
    print("QA-GATED ENSEMBLE EVALUATION")
    print("=" * 80)

    # Evaluate QA-gated ensembles
    print("\n" + "#" * 80)
    print("# QA-GATED ENSEMBLES")
    print("#" * 80)

    qa_results = evaluate_qa_ensembles(qa_folder)
    format_results_table(qa_results, ["iou_mean", "f1_mean"], "config")

    # Evaluate individual competitors
    print("\n\n" + "#" * 80)
    print("# INDIVIDUAL COMPETITORS")
    print("#" * 80)

    competitor_results = evaluate_individual_competitors(fused_split_path)
    format_results_table(competitor_results, ["jaccard_mean", "f1_mean"], "competitor")

    # Save results to CSV
    qa_results.to_csv(project_root / "qa_ensemble_evaluation_results.csv", index=False)
    competitor_results.to_csv(
        project_root / "competitor_evaluation_results.csv", index=False
    )

    print("\n\nResults saved to:")
    print("  - qa_ensemble_evaluation_results.csv")
    print("  - competitor_evaluation_results.csv")

    # Summary comparison: best QA ensemble vs best competitor (test set)
    print("\n\n" + "=" * 80)
    print("SUMMARY: TEST SET COMPARISON")
    print("=" * 80)

    qa_test = qa_results[qa_results["split"] == "test"].sort_values(
        "iou_mean", ascending=False
    )
    comp_test = competitor_results[competitor_results["split"] == "test"].sort_values(
        "jaccard_mean", ascending=False
    )

    if not qa_test.empty:
        best_qa = qa_test.iloc[0]
        print(f"\nBest QA Ensemble: {best_qa['config']}")
        print(f"  IoU: {best_qa['iou_mean']:.4f} (±{best_qa['iou_std']:.4f})")
        if "f1_mean" in best_qa:
            print(f"  F1:  {best_qa['f1_mean']:.4f} (±{best_qa['f1_std']:.4f})")

    if not comp_test.empty:
        best_comp = comp_test.iloc[0]
        print(f"\nBest Competitor: {best_comp['competitor']}")
        print(
            f"  Jaccard: {best_comp['jaccard_mean']:.4f} (±{best_comp['jaccard_std']:.4f})"
        )
        print(f"  F1:      {best_comp['f1_mean']:.4f} (±{best_comp['f1_std']:.4f})")


if __name__ == "__main__":
    main()
