"""
Detailed Cell-by-Cell Evaluation Module

This module provides detailed Jaccard score evaluation for individual cells,
storing results in a parquet format for efficient analysis and visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import click
from typing import Optional, Dict, List, Tuple
import tifffile
from skimage import measure
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

@dataclass
class CellEvaluationResult:
    """Container for individual cell evaluation results"""
    cell_id: int
    jaccard_score: float
    cell_area_gt: int
    cell_area_pred: int
    intersection_area: int
    matched_pred_label: int
    dataset_name: str
    sequence_name: str
    time_point: int
    competitor: str
    campaign_number: str
    gt_seg_path: str
    pred_seg_path: str

class DetailedCellEvaluator:
    """
    Evaluates individual cells and generates detailed parquet files with results
    """
    
    def __init__(self, dataset_df: pd.DataFrame):
        self.dataset_df = dataset_df
        
    def calculate_cell_jaccard_scores(self, 
                                    gt_image: np.ndarray, 
                                    pred_image: np.ndarray) -> List[Dict]:
        """
        Calculate Jaccard scores for each individual cell in the ground truth image.
        
        Args:
            gt_image: Ground truth segmentation image
            pred_image: Predicted segmentation image
            
        Returns:
            List of dictionaries containing cell-level evaluation results
        """
        results = []
        
        # Get all objects in ground truth (excluding background)
        gt_props = measure.regionprops(gt_image)
        
        for gt_prop in gt_props:
            gt_label = gt_prop.label
            gt_mask = (gt_image == gt_label)
            gt_area = gt_prop.area
            
            # Find overlapping labels in prediction
            overlapping_labels = np.unique(pred_image[gt_mask])
            overlapping_labels = overlapping_labels[overlapping_labels > 0]
            
            best_jaccard = 0.0
            best_pred_label = 0
            best_pred_area = 0
            best_intersection = 0
            
            # Find best matching predicted label
            for pred_label in overlapping_labels:
                pred_mask = (pred_image == pred_label)
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                
                if union > 0:
                    jaccard = intersection / union
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                        best_pred_label = pred_label
                        best_pred_area = pred_mask.sum()
                        best_intersection = intersection
            
            results.append({
                'cell_id': gt_label,
                'jaccard_score': best_jaccard,
                'cell_area_gt': gt_area,
                'cell_area_pred': best_pred_area,
                'intersection_area': best_intersection,
                'matched_pred_label': best_pred_label
            })
        
        return results
    
    def extract_metadata_from_row(self, row: pd.Series, campaign_col: str = 'campaign_number') -> Dict:
        """
        Extract metadata from dataframe row for consistent formatting.
        
        Args:
            row: Pandas series representing a row from the dataset dataframe
            campaign_col: Column name for campaign identification
            
        Returns:
            Dictionary with extracted metadata
        """
        # Extract dataset name from various possible sources
        dataset_name = row.get('dataset_name', '')
        if not dataset_name and 'composite_key' in row:
            # Try to extract from composite_key pattern: "01/BF-C2DL-MuSC/01/t0001"
            parts = str(row['composite_key']).split('/')
            if len(parts) >= 2:
                dataset_name = parts[1]
        
        # Extract sequence name
        sequence_name = row.get('sequence_name', '')
        if not sequence_name and 'composite_key' in row:
            parts = str(row['composite_key']).split('/')
            if len(parts) >= 3:
                sequence_name = parts[2]
        
        # Extract time point
        time_point = row.get('time_point', 0)
        if not time_point and 'composite_key' in row:
            parts = str(row['composite_key']).split('/')
            if len(parts) >= 4:
                time_str = parts[3]
                # Extract number from "t0001" format
                try:
                    time_point = int(time_str.replace('t', '').lstrip('0') or '0')
                except:
                    time_point = 0
        
        return {
            'dataset_name': dataset_name,
            'sequence_name': sequence_name,
            'time_point': time_point,
            'campaign_number': row.get(campaign_col, '')
        }
    
    def evaluate_competitor_detailed(self, 
                                   competitor: str, 
                                   campaign_col: str = 'campaign_number') -> pd.DataFrame:
        """
        Evaluate all cells for a given competitor and return detailed results.
        
        Args:
            competitor: Name of the competitor to evaluate
            campaign_col: Column name for campaign identification
            
        Returns:
            DataFrame with detailed cell-level evaluation results
        """
        all_results = []
        
        # Filter data for the given competitor
        if competitor not in self.dataset_df.columns:
            logging.error(f"Competitor '{competitor}' not found in dataset columns")
            return pd.DataFrame()
        
        # Filter for rows with valid ground truth and competitor data
        valid_rows = self.dataset_df[
            (self.dataset_df['gt_image'].notna()) & 
            (self.dataset_df[competitor].notna())
        ].copy()
        
        logging.info(f"Evaluating {len(valid_rows)} images for competitor '{competitor}'")
        
        processed_count = 0
        skipped_count = 0
        
        for idx, row in valid_rows.iterrows():
            try:
                # Get file paths
                gt_path = Path(row['gt_image'])
                pred_path = Path(row[competitor])
                
                if not gt_path.exists():
                    logging.warning(f"GT file not found: {gt_path}")
                    skipped_count += 1
                    continue
                    
                if not pred_path.exists():
                    logging.warning(f"Prediction file not found: {pred_path}")
                    skipped_count += 1
                    continue
                
                # Load images
                gt_image = tifffile.imread(str(gt_path))
                pred_image = tifffile.imread(str(pred_path))
                
                # Calculate cell-level scores
                cell_results = self.calculate_cell_jaccard_scores(gt_image, pred_image)
                
                # Extract metadata
                metadata = self.extract_metadata_from_row(row, campaign_col)
                
                # Add metadata to each cell result
                for cell_result in cell_results:
                    cell_result.update({
                        'competitor': competitor,
                        'gt_seg_path': str(gt_path),
                        'pred_seg_path': str(pred_path),
                        'composite_key': row.get('composite_key', ''),
                        **metadata
                    })
                
                all_results.extend(cell_results)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logging.info(f"Processed {processed_count} images...")
                
            except Exception as e:
                logging.error(f"Error processing row {idx}: {e}")
                skipped_count += 1
                continue
        
        logging.info(f"Completed evaluation: {processed_count} processed, {skipped_count} skipped")
        
        return pd.DataFrame(all_results)

@click.command()
@click.argument('dataset_parquet_path', type=click.Path(exists=True))
@click.option('--competitor', help='Specific competitor to evaluate (if not specified, evaluates all)')
@click.option('--output', '-o', help='Output parquet file path')
@click.option('--campaign-col', default='campaign_number', 
              help='Column name for campaign identification')
def evaluate_detailed(dataset_parquet_path: str, 
                     competitor: Optional[str] = None,
                     output: Optional[str] = None,
                     campaign_col: str = 'campaign_number'):
    """
    Create detailed parquet file with Jaccard scores for individual cells.
    
    This tool evaluates segmentation results at the individual cell level,
    providing detailed insights into performance for each detected object.
    
    DATASET_PARQUET_PATH: Path to the dataset dataframe parquet file
    """
    logging.info("Starting detailed cell evaluation...")
    
    try:
        # Load dataset
        df = pd.read_parquet(dataset_parquet_path)
        logging.info(f"Loaded dataset with {len(df)} rows")
        
        evaluator = DetailedCellEvaluator(df)
        
        # Determine competitors to evaluate
        if competitor:
            if competitor not in df.columns:
                logging.error(f"Competitor '{competitor}' not found in dataset")
                return
            competitors = [competitor]
        else:
            # Auto-detect competitor columns
            potential_competitors = [
                col for col in df.columns
                if col not in ['composite_key', 'raw_image', 'gt_image', campaign_col, 
                              'sequence_id', 'time_id', 'tracking_markers']
                and df[col].dtype == 'object'
                and df[col].notna().any()
            ]
            
            # Filter for columns that look like file paths
            competitors = []
            for col in potential_competitors:
                sample_val = df[col].dropna().iloc[0] if df[col].notna().any() else ""
                if isinstance(sample_val, str) and sample_val.endswith(('.tif', '.tiff')):
                    competitors.append(col)
            
            if not competitors:
                logging.error("No competitor columns found in dataset")
                return
            
            logging.info(f"Auto-detected competitors: {competitors}")
        
        all_detailed_results = []
        
        # Evaluate each competitor
        for comp in competitors:
            logging.info(f"Evaluating competitor: {comp}")
            detailed_results = evaluator.evaluate_competitor_detailed(comp, campaign_col)
            
            if not detailed_results.empty:
                all_detailed_results.append(detailed_results)
                logging.info(f"Completed {comp}: {len(detailed_results)} cells evaluated")
            else:
                logging.warning(f"No results for competitor {comp}")
        
        if not all_detailed_results:
            logging.error("No evaluation results generated")
            return
        
        # Combine all results
        final_df = pd.concat(all_detailed_results, ignore_index=True)
        
        # Generate output filename if not provided
        if not output:
            dataset_name = Path(dataset_parquet_path).stem
            if competitor:
                output = f"{dataset_name}_detailed_{competitor}.parquet"
            else:
                output = f"{dataset_name}_detailed_all_competitors.parquet"
        
        # Save results
        final_df.to_parquet(output)
        
        # Print summary
        logging.info("="*60)
        logging.info("DETAILED EVALUATION SUMMARY")
        logging.info("="*60)
        logging.info(f"Results saved to: {output}")
        logging.info(f"Total cells evaluated: {len(final_df):,}")
        logging.info(f"Competitors evaluated: {final_df['competitor'].nunique()}")
        logging.info(f"Average Jaccard score: {final_df['jaccard_score'].mean():.4f}")
        logging.info(f"Median Jaccard score: {final_df['jaccard_score'].median():.4f}")
        logging.info(f"Standard deviation: {final_df['jaccard_score'].std():.4f}")
        
        # Per-competitor summary
        competitor_summary = final_df.groupby('competitor')['jaccard_score'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(4)
        logging.info("\nPer-Competitor Summary:")
        logging.info(competitor_summary.to_string())
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

if __name__ == '__main__':
    evaluate_detailed()
