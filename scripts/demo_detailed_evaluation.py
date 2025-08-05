"""
Quick demonstration notebook for detailed cell evaluation tools

This script demonstrates how to use the detailed evaluation tools
and provides examples of basic analysis workflows.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def demo_detailed_evaluation():
    """
    Demonstrate the detailed evaluation workflow
    """
    print("=" * 60)
    print("DETAILED CELL EVALUATION DEMO")
    print("=" * 60)

    # Example 1: Load and examine detailed results
    print("\n1. Loading detailed evaluation results...")

    # This would normally load your actual results file
    demo_file = "detailed_BF-C2DL-MuSC.parquet"

    if Path(demo_file).exists():
        df = pd.read_parquet(demo_file)
        print(f"   Loaded {len(df):,} cell evaluations")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Competitors: {df['competitor'].unique()}")

        # Basic statistics
        print("\n2. Basic Statistics:")
        print(f"   Average Jaccard Score: {df['jaccard_score'].mean():.4f}")
        print(f"   Median Jaccard Score: {df['jaccard_score'].median():.4f}")
        print(f"   Standard Deviation: {df['jaccard_score'].std():.4f}")

        # Competitor comparison
        print("\n3. Competitor Comparison:")
        comp_stats = (
            df.groupby("competitor")["jaccard_score"]
            .agg(["count", "mean", "std"])
            .round(4)
        )
        print(comp_stats)

        # Cell size analysis
        if "cell_area_gt" in df.columns:
            print("\n4. Cell Size Analysis:")
            correlation = df["cell_area_gt"].corr(df["jaccard_score"])
            print(
                f"   Correlation between cell size and Jaccard score: {correlation:.4f}"
            )

            # Size bins
            df["size_bin"] = pd.cut(
                df["cell_area_gt"],
                bins=[0, 100, 500, 1000, float("inf")],
                labels=["Small", "Medium", "Large", "Very Large"],
            )
            size_performance = df.groupby("size_bin")["jaccard_score"].mean()
            print("   Performance by cell size:")
            for size, score in size_performance.items():
                print(f"     {size}: {score:.4f}")

    else:
        print(f"   Demo file {demo_file} not found.")
        print("   To create it, run:")
        print(
            f"     python detailed_evaluation.py dataframes/BF-C2DL-MuSC_dataset_dataframe.parquet --output {demo_file}"
        )


def demo_analysis_workflow():
    """
    Demonstrate analysis workflow
    """
    print("\n" + "=" * 60)
    print("ANALYSIS WORKFLOW DEMO")
    print("=" * 60)

    print("\nStep-by-step workflow for detailed evaluation:")
    print("\n1. Run detailed evaluation:")
    print(
        "   python detailed_evaluation.py dataframes/BF-C2DL-MuSC_dataset_dataframe.parquet"
    )

    print("\n2. Analyze results:")
    print(
        "   python analyze_detailed_results.py detailed_BF-C2DL-MuSC_all_competitors.parquet"
    )

    print("\n3. For batch processing:")
    print("   python run_all_detailed_evaluations.py")

    print("\n4. Integration with existing workflow:")
    print(
        "   python evaluation.py evaluate-competitor dataframes/BF-C2DL-MuSC_dataset_dataframe.parquet --detailed"
    )


def create_sample_visualization():
    """
    Create a sample visualization with synthetic data
    """
    print("\n" + "=" * 60)
    print("SAMPLE VISUALIZATION")
    print("=" * 60)

    # Create synthetic data for demonstration
    np.random.seed(42)
    n_cells = 1000

    competitors = ["Competitor_A", "Competitor_B", "Competitor_C"]
    data = []

    for comp in competitors:
        n_comp_cells = n_cells // len(competitors)

        if comp == "Competitor_A":  # Good performer
            scores = np.random.beta(8, 2, n_comp_cells)  # Skewed towards high scores
        elif comp == "Competitor_B":  # Average performer
            scores = np.random.beta(4, 4, n_comp_cells)  # Normal distribution
        else:  # Poor performer
            scores = np.random.beta(2, 6, n_comp_cells)  # Skewed towards low scores

        cell_areas = np.random.lognormal(4, 1, n_comp_cells)

        for i in range(n_comp_cells):
            data.append(
                {
                    "competitor": comp,
                    "jaccard_score": scores[i],
                    "cell_area_gt": int(cell_areas[i]),
                    "cell_id": i + 1,
                }
            )

    df_demo = pd.DataFrame(data)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Score distribution by competitor
    sns.boxplot(data=df_demo, x="competitor", y="jaccard_score", ax=axes[0, 0])
    axes[0, 0].set_title("Jaccard Score Distribution by Competitor")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Overall score histogram
    axes[0, 1].hist(df_demo["jaccard_score"], bins=30, alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Jaccard Score")
    axes[0, 1].set_ylabel("Number of Cells")
    axes[0, 1].set_title("Overall Jaccard Score Distribution")

    # 3. Cell size vs score
    for comp in competitors:
        comp_data = df_demo[df_demo["competitor"] == comp]
        axes[1, 0].scatter(
            comp_data["cell_area_gt"], comp_data["jaccard_score"], label=comp, alpha=0.6
        )
    axes[1, 0].set_xlabel("Cell Area")
    axes[1, 0].set_ylabel("Jaccard Score")
    axes[1, 0].set_title("Cell Size vs Performance")
    axes[1, 0].legend()
    axes[1, 0].set_xscale("log")

    # 4. Summary statistics table
    axes[1, 1].axis("off")
    summary_stats = (
        df_demo.groupby("competitor")["jaccard_score"]
        .agg(["mean", "median", "std"])
        .round(3)
    )
    table_data = []
    for comp in summary_stats.index:
        table_data.append(
            [
                comp,
                summary_stats.loc[comp, "mean"],
                summary_stats.loc[comp, "median"],
                summary_stats.loc[comp, "std"],
            ]
        )

    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=["Competitor", "Mean", "Median", "Std"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title("Summary Statistics")

    plt.tight_layout()

    # Save the plot
    output_file = "demo_detailed_evaluation_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSample visualization saved as: {output_file}")
    plt.show()

    return df_demo


def main():
    """Main demo function"""
    print("Welcome to the Detailed Cell Evaluation Demo!")
    print("This demonstrates the new detailed evaluation capabilities.")

    # Run demonstrations
    demo_detailed_evaluation()
    demo_analysis_workflow()

    # Create sample visualization
    _ = create_sample_visualization()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey benefits of detailed evaluation:")
    print("✅ Individual cell-level analysis")
    print("✅ Efficient parquet storage format")
    print("✅ Comprehensive statistical analysis")
    print("✅ Rich visualization capabilities")
    print("✅ Integration with existing workflow")

    print("\nNext steps:")
    print("1. Run detailed evaluation on your datasets")
    print("2. Use analyze_detailed_results.py for insights")
    print("3. Integrate with your existing evaluation pipeline")


if __name__ == "__main__":
    main()
