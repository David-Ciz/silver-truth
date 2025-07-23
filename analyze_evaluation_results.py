#!/usr/bin/env python3
"""
Script to analyze and summarize evaluation results from all datasets.

This script finds all evaluation result CSV files and creates a comprehensive
summary analysis including statistics across datasets and competitors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
RESULTS_DIR = "evaluation_results"
RESULTS_PATTERN = "evaluation_results_*.csv"
OUTPUT_DIR = "analysis_results"

def load_all_results():
    """Load all evaluation result CSV files."""
    results_pattern = f"{RESULTS_DIR}/{RESULTS_PATTERN}"
    result_files = glob.glob(results_pattern)
    
    if not result_files:
        logging.warning(f"No result files found matching pattern: {results_pattern}")
        return None
    
    logging.info(f"Found {len(result_files)} result files to analyze")
    
    all_results = []
    
    for result_file in result_files:
        try:
            # Extract dataset name from filename
            filename = Path(result_file).stem
            dataset_name = filename.replace("evaluation_results_", "")
            
            df = pd.read_csv(result_file)
            df['dataset'] = dataset_name
            all_results.append(df)
            
            logging.info(f"Loaded {len(df)} results from {dataset_name}")
            
        except Exception as e:
            logging.error(f"Failed to load {result_file}: {e}")
    
    if not all_results:
        logging.error("No valid result files were loaded")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    logging.info(f"Combined total: {len(combined_df)} evaluation records")
    
    return combined_df

def create_summary_statistics(df):
    """Create summary statistics from the combined results."""
    logging.info("Creating summary statistics...")
    
    summary_stats = {}
    
    # Overall statistics
    summary_stats['overall'] = {
        'total_evaluations': len(df),
        'unique_datasets': df['dataset'].nunique(),
        'unique_competitors': df['competitor'].nunique(),
        'mean_jaccard': df['jaccard_score'].mean(),
        'median_jaccard': df['jaccard_score'].median(),
        'std_jaccard': df['jaccard_score'].std(),
        'min_jaccard': df['jaccard_score'].min(),
        'max_jaccard': df['jaccard_score'].max()
    }
    
    # Per-dataset statistics
    dataset_stats = df.groupby('dataset')['jaccard_score'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    summary_stats['per_dataset'] = dataset_stats
    
    # Per-competitor statistics
    competitor_stats = df.groupby('competitor')['jaccard_score'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    summary_stats['per_competitor'] = competitor_stats
    
    # Top performers
    top_competitors = competitor_stats.sort_values('mean', ascending=False)
    summary_stats['top_competitors'] = top_competitors.head(10)
    
    return summary_stats

def create_visualizations(df, output_dir):
    """Create visualizations of the evaluation results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        logging.warning(f"Visualization libraries not available: {e}")
        logging.warning("Skipping visualizations. Install matplotlib and seaborn to enable visualizations.")
        return
    
    logging.info("Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Distribution of Jaccard scores
    plt.figure(figsize=(10, 6))
    plt.hist(df['jaccard_score'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Jaccard Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Jaccard Scores Across All Evaluations')
    plt.grid(True, alpha=0.3)
    plt.savefig(viz_dir / "jaccard_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot by dataset
    plt.figure(figsize=(15, 8))
    df_sorted = df.copy()
    dataset_means = df.groupby('dataset')['jaccard_score'].mean().sort_values(ascending=False)
    df_sorted['dataset'] = pd.Categorical(df_sorted['dataset'], categories=dataset_means.index, ordered=True)
    
    sns.boxplot(data=df_sorted, x='dataset', y='jaccard_score')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Dataset')
    plt.ylabel('Jaccard Score')
    plt.title('Jaccard Score Distribution by Dataset')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "jaccard_by_dataset.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top competitors performance
    top_competitors = df.groupby('competitor')['jaccard_score'].mean().sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    top_competitors.plot(kind='bar')
    plt.xlabel('Competitor')
    plt.ylabel('Mean Jaccard Score')
    plt.title('Top 15 Competitors by Mean Jaccard Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "top_competitors.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap: Dataset vs Competitor (if reasonable size)
    if df['dataset'].nunique() <= 20 and df['competitor'].nunique() <= 30:
        pivot_table = df.pivot_table(values='jaccard_score', index='dataset', columns='competitor', aggfunc='mean')
        
        plt.figure(figsize=(20, 12))
        sns.heatmap(pivot_table, annot=False, cmap='viridis', cbar_kws={'label': 'Mean Jaccard Score'})
        plt.xlabel('Competitor')
        plt.ylabel('Dataset')
        plt.title('Mean Jaccard Score: Dataset vs Competitor')
        plt.tight_layout()
        plt.savefig(viz_dir / "dataset_competitor_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Visualizations saved to: {viz_dir}")

def save_summary_report(summary_stats, df, output_dir):
    """Save a comprehensive summary report."""
    logging.info("Creating summary report...")
    
    report_path = Path(output_dir) / "evaluation_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CELL TRACKING EVALUATION SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*20 + "\n")
        for key, value in summary_stats['overall'].items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Top datasets
        f.write("TOP 10 DATASETS BY MEAN JACCARD SCORE\n")
        f.write("-"*40 + "\n")
        top_datasets = summary_stats['per_dataset'].sort_values('mean', ascending=False).head(10)
        f.write(top_datasets.to_string())
        f.write("\n\n")
        
        # Top competitors
        f.write("TOP 10 COMPETITORS BY MEAN JACCARD SCORE\n")
        f.write("-"*42 + "\n")
        f.write(summary_stats['top_competitors'].to_string())
        f.write("\n\n")
        
        # Dataset details
        f.write("DETAILED DATASET STATISTICS\n")
        f.write("-"*28 + "\n")
        f.write(summary_stats['per_dataset'].to_string())
        f.write("\n\n")
        
        # Competitor details (showing top 20 to avoid too long report)
        f.write("TOP 20 COMPETITOR STATISTICS\n")
        f.write("-"*28 + "\n")
        top_20_competitors = summary_stats['per_competitor'].sort_values('mean', ascending=False).head(20)
        f.write(top_20_competitors.to_string())
        f.write("\n")
    
    logging.info(f"Summary report saved to: {report_path}")

def main():
    """Main function to analyze all evaluation results."""
    # Load all results
    df = load_all_results()
    if df is None:
        logging.error("No evaluation results found. Make sure to run 'python run_all_evaluations.py' first.")
        return
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Create summary statistics
    summary_stats = create_summary_statistics(df)
    
    # Save detailed CSV with all combined results
    combined_results_path = output_dir / "combined_evaluation_results.csv"
    df.to_csv(combined_results_path, index=False, float_format="%.6f")
    logging.info(f"Combined results saved to: {combined_results_path}")
    
    # Save summary statistics as CSV
    summary_stats['per_dataset'].to_csv(output_dir / "dataset_summary.csv")
    summary_stats['per_competitor'].to_csv(output_dir / "competitor_summary.csv")
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    # Create summary report
    save_summary_report(summary_stats, df, output_dir)
    
    # Print key findings
    logging.info(f"\n{'='*50}")
    logging.info("KEY FINDINGS")
    logging.info(f"{'='*50}")
    logging.info(f"Total evaluations processed: {summary_stats['overall']['total_evaluations']}")
    logging.info(f"Datasets analyzed: {summary_stats['overall']['unique_datasets']}")
    logging.info(f"Competitors analyzed: {summary_stats['overall']['unique_competitors']}")
    logging.info(f"Overall mean Jaccard score: {summary_stats['overall']['mean_jaccard']:.4f}")
    logging.info(f"Overall median Jaccard score: {summary_stats['overall']['median_jaccard']:.4f}")
    
    best_competitor = summary_stats['per_competitor'].sort_values('mean', ascending=False).index[0]
    best_score = summary_stats['per_competitor'].sort_values('mean', ascending=False)['mean'].iloc[0]
    logging.info(f"Best performing competitor: {best_competitor} (mean: {best_score:.4f})")
    
    best_dataset = summary_stats['per_dataset'].sort_values('mean', ascending=False).index[0]
    best_dataset_score = summary_stats['per_dataset'].sort_values('mean', ascending=False)['mean'].iloc[0]
    logging.info(f"Best performing dataset: {best_dataset} (mean: {best_dataset_score:.4f})")
    
    logging.info(f"\nAll analysis results saved to: {output_dir}")
    logging.info("Analysis completed successfully!")

if __name__ == "__main__":
    main()
