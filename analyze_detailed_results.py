"""
Analysis module for detailed cell evaluation results

Provides statistical analysis and visualization tools for detailed cell-level
Jaccard score evaluations stored in parquet format.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path
import logging
from typing import Optional, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class DetailedResultsAnalyzer:
    """
    Analyzer for detailed cell evaluation results
    """
    
    def __init__(self, detailed_results_df: pd.DataFrame):
        self.df = detailed_results_df
        self._validate_data()
    
    def _validate_data(self):
        """Validate that the dataframe has required columns"""
        required_cols = ['cell_id', 'jaccard_score', 'competitor', 'cell_area_gt']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if self.df.empty:
            raise ValueError("DataFrame is empty")
        
        logging.info(f"Data validation passed. Dataset contains {len(self.df)} cell evaluations")
    
    def summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        stats = {}
        
        # Overall statistics
        stats['total_cells'] = len(self.df)
        stats['unique_competitors'] = self.df['competitor'].nunique()
        stats['competitors'] = sorted(self.df['competitor'].unique())
        
        if 'dataset_name' in self.df.columns:
            stats['datasets'] = sorted(self.df['dataset_name'].unique())
            stats['unique_datasets'] = self.df['dataset_name'].nunique()
        
        # Jaccard score statistics
        jaccard_stats = self.df['jaccard_score'].describe()
        stats['jaccard_mean'] = jaccard_stats['mean']
        stats['jaccard_median'] = jaccard_stats['50%']
        stats['jaccard_std'] = jaccard_stats['std']
        stats['jaccard_min'] = jaccard_stats['min']
        stats['jaccard_max'] = jaccard_stats['max']
        
        # Zero score analysis
        zero_scores = (self.df['jaccard_score'] == 0).sum()
        stats['zero_scores_count'] = zero_scores
        stats['zero_scores_percentage'] = (zero_scores / len(self.df)) * 100
        
        # High score analysis (>0.7)
        high_scores = (self.df['jaccard_score'] > 0.7).sum()
        stats['high_scores_count'] = high_scores
        stats['high_scores_percentage'] = (high_scores / len(self.df)) * 100
        
        return stats
    
    def competitor_comparison(self) -> pd.DataFrame:
        """
        Compare performance across competitors
        
        Returns:
            DataFrame with competitor comparison statistics
        """
        competitor_stats = self.df.groupby('competitor')['jaccard_score'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(4)
        
        # Add percentage of zero scores
        zero_scores = self.df[self.df['jaccard_score'] == 0].groupby('competitor').size()
        competitor_stats['zero_scores'] = zero_scores.fillna(0).astype(int)
        competitor_stats['zero_percent'] = (
            competitor_stats['zero_scores'] / competitor_stats['count'] * 100
        ).round(2)
        
        # Add percentage of high scores (>0.7)
        high_scores = self.df[self.df['jaccard_score'] > 0.7].groupby('competitor').size()
        competitor_stats['high_scores'] = high_scores.fillna(0).astype(int)
        competitor_stats['high_percent'] = (
            competitor_stats['high_scores'] / competitor_stats['count'] * 100
        ).round(2)
        
        return competitor_stats.sort_values('mean', ascending=False)
    
    def cell_size_analysis(self) -> Dict:
        """
        Analyze relationship between cell size and Jaccard scores
        
        Returns:
            Dictionary with cell size analysis results
        """
        # Define size bins
        size_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
        size_labels = ['Tiny(<50)', 'Small(50-100)', 'Medium(100-200)', 
                      'Large(200-500)', 'Very Large(500-1000)', 'Huge(>1000)']
        
        self.df['size_category'] = pd.cut(
            self.df['cell_area_gt'], 
            bins=size_bins, 
            labels=size_labels, 
            include_lowest=True
        )
        
        size_analysis = self.df.groupby('size_category')['jaccard_score'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(4)
        
        return {
            'size_stats': size_analysis,
            'correlation': self.df['cell_area_gt'].corr(self.df['jaccard_score'])
        }
    
    def print_summary_report(self):
        """Print comprehensive summary report to console"""
        print("="*80)
        print("DETAILED CELL EVALUATION ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        stats = self.summary_statistics()
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   Total cells evaluated: {stats['total_cells']:,}")
        print(f"   Unique competitors: {stats['unique_competitors']}")
        print(f"   Competitors: {', '.join(stats['competitors'])}")
        
        if 'datasets' in stats:
            print(f"   Datasets: {', '.join(stats['datasets'])}")
        
        print(f"\n🎯 JACCARD SCORE STATISTICS")
        print(f"   Mean: {stats['jaccard_mean']:.4f}")
        print(f"   Median: {stats['jaccard_median']:.4f}")
        print(f"   Standard Deviation: {stats['jaccard_std']:.4f}")
        print(f"   Range: {stats['jaccard_min']:.4f} - {stats['jaccard_max']:.4f}")
        
        print(f"\n📈 PERFORMANCE DISTRIBUTION")
        print(f"   Zero scores: {stats['zero_scores_count']:,} ({stats['zero_scores_percentage']:.1f}%)")
        print(f"   High scores (>0.7): {stats['high_scores_count']:,} ({stats['high_scores_percentage']:.1f}%)")
        
        # Competitor comparison
        print(f"\n🏆 COMPETITOR COMPARISON")
        competitor_stats = self.competitor_comparison()
        print(competitor_stats.to_string())
        
        # Cell size analysis
        if 'cell_area_gt' in self.df.columns:
            print(f"\n📏 CELL SIZE ANALYSIS")
            size_analysis = self.cell_size_analysis()
            print("Performance by cell size:")
            print(size_analysis['size_stats'].to_string())
            print(f"\nCorrelation between cell size and Jaccard score: {size_analysis['correlation']:.4f}")
    
    def create_visualizations(self, output_dir: str = "detailed_analysis"):
        """
        Create comprehensive visualizations
        
        Args:
            output_dir: Directory to save visualization files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Jaccard score distribution histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall distribution
        ax1.hist(self.df['jaccard_score'], bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Jaccard Score')
        ax1.set_ylabel('Number of Cells')
        ax1.set_title('Distribution of Jaccard Scores')
        ax1.axvline(self.df['jaccard_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["jaccard_score"].mean():.3f}')
        ax1.axvline(self.df['jaccard_score'].median(), color='orange', linestyle='--', 
                   label=f'Median: {self.df["jaccard_score"].median():.3f}')
        ax1.legend()
        
        # Cumulative distribution
        sorted_scores = np.sort(self.df['jaccard_score'])
        y_vals = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax2.plot(sorted_scores, y_vals)
        ax2.set_xlabel('Jaccard Score')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution of Jaccard Scores')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'jaccard_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Competitor comparison boxplot
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.df, y='competitor', x='jaccard_score', orient='h')
        plt.title('Jaccard Score Distribution by Competitor')
        plt.xlabel('Jaccard Score')
        plt.tight_layout()
        plt.savefig(output_path / 'competitor_comparison_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Competitor comparison violin plot
        plt.figure(figsize=(14, 8))
        sns.violinplot(data=self.df, x='competitor', y='jaccard_score')
        plt.xticks(rotation=45)
        plt.title('Jaccard Score Distribution by Competitor (Violin Plot)')
        plt.ylabel('Jaccard Score')
        plt.tight_layout()
        plt.savefig(output_path / 'competitor_comparison_violin.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Cell size vs Jaccard score scatter plot
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with different colors for competitors
        competitors = self.df['competitor'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(competitors)))
        
        for i, comp in enumerate(competitors):
            comp_data = self.df[self.df['competitor'] == comp]
            plt.scatter(comp_data['cell_area_gt'], comp_data['jaccard_score'], 
                       alpha=0.6, label=comp, color=colors[i], s=20)
        
        plt.xlabel('Cell Area (Ground Truth)')
        plt.ylabel('Jaccard Score')
        plt.title('Cell Size vs. Jaccard Score by Competitor')
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'size_vs_jaccard_by_competitor.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Performance heatmap by competitor and dataset (if available)
        if 'dataset_name' in self.df.columns and self.df['dataset_name'].nunique() > 1:
            pivot_data = self.df.groupby(['competitor', 'dataset_name'])['jaccard_score'].mean().unstack()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r')
            plt.title('Average Jaccard Score by Competitor and Dataset')
            plt.ylabel('Competitor')
            plt.xlabel('Dataset')
            plt.tight_layout()
            plt.savefig(output_path / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Cell size distribution by competitor
        if 'size_category' in self.df.columns:
            plt.figure(figsize=(12, 8))
            size_comp = self.df.groupby(['competitor', 'size_category'])['jaccard_score'].mean().unstack()
            size_comp.plot(kind='bar', ax=plt.gca())
            plt.title('Average Jaccard Score by Competitor and Cell Size')
            plt.xlabel('Competitor')
            plt.ylabel('Average Jaccard Score')
            plt.xticks(rotation=45)
            plt.legend(title='Cell Size Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(output_path / 'performance_by_size_category.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logging.info(f"Visualizations saved to: {output_path}")
        return output_path

@click.command()
@click.argument('detailed_results_path', type=click.Path(exists=True))
@click.option('--output-dir', default='detailed_analysis', 
              help='Directory for analysis outputs')
@click.option('--save-summary', is_flag=True,
              help='Save summary statistics to CSV file')
def analyze_detailed_results(detailed_results_path: str, 
                           output_dir: str,
                           save_summary: bool = False):
    """
    Analyze detailed cell evaluation results and generate comprehensive reports.
    
    DETAILED_RESULTS_PATH: Path to the detailed evaluation parquet file
    """
    try:
        logging.info(f"Loading detailed results from: {detailed_results_path}")
        df = pd.read_parquet(detailed_results_path)
        
        analyzer = DetailedResultsAnalyzer(df)
        
        # Print summary report
        analyzer.print_summary_report()
        
        # Create visualizations
        output_path = analyzer.create_visualizations(output_dir)
        
        # Save summary statistics if requested
        if save_summary:
            competitor_stats = analyzer.competitor_comparison()
            summary_file = Path(output_dir) / 'competitor_summary.csv'
            competitor_stats.to_csv(summary_file)
            logging.info(f"Summary statistics saved to: {summary_file}")
        
        logging.info("Analysis completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise

if __name__ == '__main__':
    analyze_detailed_results()
