import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import tifffile
import numpy as np

from src.metrics.metrics import calculate_qa_jaccard_score

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_qa_evaluation(
    qa_dataframe_path: Path,
    ground_truth_dataframe_path: Path,
    competitor: Optional[str] = None,
    output: Optional[Path] = None,
    visualize: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates QA cropped images against ground truth using Jaccard index.
    
    This function is specifically designed to work with the stacked TIFF files
    created by the QA data preprocessor, where:
    - Channel 0: Raw image (cropped or full)
    - Channel 1: Cell mask (binary: 0 or 255)
    
    Args:
        qa_dataframe_path: Path to the QA dataframe (parquet file with cropped images metadata)
        ground_truth_dataframe_path: Path to the original dataset dataframe with GT information
        competitor: Specific competitor to evaluate (optional)
        output: Path to save results as CSV (optional)
        visualize: Generate visualization (placeholder)
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Load QA dataframe
        qa_df = pd.read_parquet(qa_dataframe_path)
        logging.info(f"=== EVALUATING QA DATASET ===")
        logging.info(f"Total QA images: {len(qa_df)}")
        
        # Load original dataset dataframe to get GT information
        from src.data_processing.utils.dataset_dataframe_creation import (
            load_dataframe_from_parquet_with_metadata,
        )
        gt_df = load_dataframe_from_parquet_with_metadata(str(ground_truth_dataframe_path))
        
        # Check for required columns in QA dataframe
        required_columns = ['cell_id', 'stacked_path', 'original_image_key', 'campaign_number', 'competitor', 'label']
        missing_columns = [col for col in required_columns if col not in qa_df.columns]
        if missing_columns:
            logging.error(f"Missing required columns in QA dataframe: {missing_columns}")
            return {}
            
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return {}
    except Exception as e:
        logging.error(f"Error loading dataframes: {e}")
        return {}

    # Filter by competitor if specified
    competitors = qa_df['competitor'].unique()
    if competitor:
        if competitor not in competitors:
            logging.error(f"Competitor '{competitor}' not found in QA data. Available: {competitors}")
            return {}
        qa_df = qa_df[qa_df['competitor'] == competitor].copy()
        competitors = [competitor]
        logging.info(f"Evaluating specific competitor: {competitor}")
    else:
        logging.info(f"Evaluating all competitors: {competitors}")

    # Get campaigns
    campaigns = sorted(qa_df['campaign_number'].unique())
    logging.info(f"Found {len(campaigns)} campaigns: {campaigns}")

    # Initialize results structures
    all_results = {comp: {camp: {} for camp in campaigns} for comp in competitors}
    per_image_averages = {comp: {camp: {} for camp in campaigns} for comp in competitors}
    per_campaign_averages = {comp: {} for comp in competitors}
    overall_averages = {}

    # Processing loop
    for comp in competitors:
        logging.info(f"--- Processing Competitor: {comp} ---")
        competitor_all_scores = []
        
        comp_df = qa_df[qa_df['competitor'] == comp].copy()
        
        for campaign in campaigns:
            logging.info(f"  Processing Campaign: {campaign}")
            campaign_df = comp_df[comp_df['campaign_number'] == campaign].copy()
            campaign_scores = []
            
            processed_count = 0
            skipped_count = 0
            
            for index, row in campaign_df.iterrows():
                cell_id = row['cell_id']
                stacked_path = Path(row['stacked_path'])
                original_image_key = row['original_image_key']
                label = row['label']
                
                if not stacked_path.exists():
                    logging.warning(f"    {cell_id}: Stacked image not found: {stacked_path}")
                    skipped_count += 1
                    continue
                
                # Find corresponding GT image
                # Convert QA image key format (t002) to GT format (01_0002.tif)
                if original_image_key.startswith('t'):
                    time_part = original_image_key[1:]  # Remove 't' prefix (e.g., "002")
                    # Pad to 4 digits to match GT format
                    time_part_padded = time_part.zfill(4)  # "002" -> "0002"
                    expected_gt_key = f"{campaign}_{time_part_padded}.tif"
                    gt_row = gt_df[gt_df['composite_key'] == expected_gt_key]
                else:
                    # Fallback to original logic if format is different
                    gt_row = gt_df[
                        (gt_df['composite_key'].str.contains(original_image_key, na=False)) &
                        (gt_df['campaign_number'] == campaign)
                    ]
                
                if gt_row.empty:
                    logging.warning(f"    {cell_id}: No GT found for {original_image_key} in campaign {campaign}")
                    skipped_count += 1
                    continue
                    
                gt_path_str = gt_row.iloc[0]['gt_image']
                if pd.isna(gt_path_str) or not Path(gt_path_str).exists():
                    logging.warning(f"    {cell_id}: GT file not found: {gt_path_str}")
                    skipped_count += 1
                    continue
                
                try:
                    # Load stacked image (2 channels: raw + mask)
                    stacked_image = tifffile.imread(stacked_path)
                    if stacked_image.ndim != 3 or stacked_image.shape[0] != 2:
                        logging.warning(f"    {cell_id}: Invalid stacked image format. Expected 2 channels, got shape: {stacked_image.shape}")
                        skipped_count += 1
                        continue
                    
                    # Extract the mask (channel 1, convert from 0-255 to 0-1)
                    predicted_mask = (stacked_image[1] > 0).astype(np.uint8)
                    
                    # Load GT image
                    gt_image = tifffile.imread(gt_path_str)
                    
                    # Calculate Jaccard score for this specific cell crop
                    jaccard_score = calculate_qa_jaccard_score(
                        gt_image, predicted_mask, label, 
                        original_image_key, campaign, row
                    )
                    
                    if jaccard_score is not None:
                        # Store results
                        if cell_id not in all_results[comp][campaign]:
                            all_results[comp][campaign][cell_id] = {}
                        all_results[comp][campaign][cell_id][label] = jaccard_score
                        
                        campaign_scores.append(jaccard_score)
                        processed_count += 1
                    else:
                        skipped_count += 1
                        
                except Exception as e:
                    logging.error(f"    {cell_id}: Error processing: {e}")
                    skipped_count += 1
            
            logging.info(f"    Campaign '{campaign}': Processed {processed_count}, Skipped {skipped_count} images")
            
            # Campaign average
            if campaign_scores:
                campaign_avg = sum(campaign_scores) / len(campaign_scores)
                per_campaign_averages[comp][campaign] = campaign_avg
                competitor_all_scores.extend(campaign_scores)
                logging.info(f"    Campaign '{campaign}' avg Jaccard: {campaign_avg:.4f} (from {len(campaign_scores)} scores)")
            else:
                per_campaign_averages[comp][campaign] = float("nan")
                logging.warning(f"    No valid scores for campaign '{campaign}'")
        
        # Overall average for competitor
        if competitor_all_scores:
            overall_avg = sum(competitor_all_scores) / len(competitor_all_scores)
            overall_averages[comp] = overall_avg
            logging.info(f"  Overall avg Jaccard for '{comp}': {overall_avg:.4f} (from {len(competitor_all_scores)} scores)")
        else:
            overall_averages[comp] = float("nan")
            logging.warning(f"  No valid scores for competitor '{comp}'")

    # Print summary
    logging.info("--- QA Evaluation Summary ---")
    valid_competitors = sum(1 for comp in competitors if not pd.isna(overall_averages.get(comp, float('nan'))))
    logging.info(f"Competitors with valid scores: {valid_competitors}/{len(competitors)}")
    
    for comp in competitors:
        avg = overall_averages.get(comp, float('nan'))
        if not pd.isna(avg):
            logging.info(f"  {comp}: {avg:.4f}")
        else:
            logging.info(f"  {comp}: No valid scores")

    # Save results to CSV if requested
    if output:
        save_qa_results_to_csv(
            all_results, per_campaign_averages, overall_averages, 
            qa_df, output
        )

    if visualize:
        logging.warning("Visualization not implemented for QA evaluation")

    logging.info("QA Evaluation completed.")

    return {
        "all_results": all_results,
        "per_campaign_averages": per_campaign_averages,
        "overall_averages": overall_averages,
    }


def save_qa_results_to_csv(all_results, per_campaign_averages, overall_averages, qa_df, output_path):
    """Save QA evaluation results to CSV file."""
    try:
        output_data = []
        
        for comp, campaigns_data in all_results.items():
            for campaign, cells_data in campaigns_data.items():
                for cell_id, labels_data in cells_data.items():
                    camp_avg = per_campaign_averages.get(comp, {}).get(campaign, float("nan"))
                    overall_avg = overall_averages.get(comp, float("nan"))
                    
                    # Get original metadata from QA dataframe
                    qa_row = qa_df[qa_df['cell_id'] == cell_id]
                    if not qa_row.empty:
                        qa_info = qa_row.iloc[0]
                        
                        for label, score in labels_data.items():
                            output_data.append({
                                "competitor": comp,
                                "campaign": campaign,
                                "cell_id": cell_id,
                                "original_image_key": qa_info['original_image_key'],
                                "label": label,
                                "jaccard_score": score,
                                "campaign_average": camp_avg,
                                "overall_competitor_average": overall_avg,
                                "stacked_path": qa_info['stacked_path']
                            })
        
        if output_data:
            results_df = pd.DataFrame(output_data)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False, float_format="%.6f")
            logging.info(f"QA evaluation results saved to {output_path}")
        else:
            logging.warning("No results data to save to CSV")
            
    except Exception as e:
        logging.error(f"Failed to save QA results to CSV: {e}")
