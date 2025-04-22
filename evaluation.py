import logging
import os
import re
from pathlib import Path

import click
import tifffile
import tqdm
from src.data_processing.label_synchronizer import verify_synchronization, synchronize_labels_with_tracking_markers, \
    process_segmentation_folders, verify_folder_synchronization_logic
import pandas as pd

from src.data_processing.utils.dataset_dataframe_creation import load_dataframe_from_parquet_with_metadata
from src.metrics.metrics import calculate_jaccard_scores
from src.metrics.utils import print_results

# Constants
RAW_DATA_FOLDERS = {"01", "02"}
GT_FOLDER_FIRST = "01_GT"
GT_FOLDER_SECOND = "02_GT"
SEG_FOLDER = "SEG"
TRA_FOLDER = "TRA"
RES_FOLDER_FIRST = "01_RES"
RES_FOLDER_SECOND = "02_RES"


@click.command()
@click.argument('dataset_dataframe_path', type=click.Path(exists=True))
@click.option('--competitor')
@click.option('--output', '-o', type=click.Path(), help="Path to save results as CSV")
@click.option('--visualize', '-v', is_flag=True, help="Generate visualization of results")
@click.option('--campaign-col', default='campaign_number', help="Column name that identifies the campaign")
def evaluate_competitor(dataset_dataframe_path: str | Path, competitor: str = None,
                        output: str = None, visualize: bool = False, campaign_col: str = 'campaign_number'):
    """
    This script evaluates a competitor's results against the ground truth.
    Outputs the Jaccard index for each segmentation, with per-label scores,
    image averages, and campaign averages.

    DATASET_DATAFRAME_PATH: Path to the dataset dataframe file
    competitor: Competitor name to evaluate if not all competitors are to be evaluated
    """
    df = load_dataframe_from_parquet_with_metadata(dataset_dataframe_path)
    competitor_columns = df.attrs['competitor_columns']

    if competitor:
        if competitor not in competitor_columns:
            logging.error(f"Competitor {competitor} not found in dataset dataframe. "
                          f"Known competitors: {competitor_columns}")
            return
        competitor_columns = [competitor]

    # Filter rows where the ground truth image is not None
    filtered_df = df[df['gt_image'].notnull()]

    # Check if campaign column exists, if not try to extract from composite_key
    if campaign_col not in filtered_df.columns:
        logging.warning(f"Campaign column '{campaign_col}' not found. Attempting to extract from composite_key.")
        # Assuming composite_key format might be "campaign_name/image_id" or similar
        filtered_df[campaign_col] = filtered_df['composite_key'].apply(
            lambda x: x.split('/')[0] if '/' in str(x) else 'unknown_campaign'
        )

    # Get unique campaigns
    campaigns = filtered_df[campaign_col].unique()

    # Create data structures to store results
    all_results = {}  # comp -> campaign -> image -> label -> score
    per_image_averages = {}  # comp -> campaign -> image -> avg_score
    per_campaign_averages = {}  # comp -> campaign -> avg_score
    overall_averages = {}  # comp -> avg_score
    all_labels = set()

    # Process each competitor
    for comp in competitor_columns:
        all_results[comp] = {}
        per_image_averages[comp] = {}
        per_campaign_averages[comp] = {}
        all_label_scores = []

        # Process each campaign
        for campaign in campaigns:
            campaign_df = filtered_df[filtered_df[campaign_col] == campaign]
            all_results[comp][campaign] = {}
            per_image_averages[comp][campaign] = {}
            campaign_label_scores = []

            # Process each image in this campaign
            for _, row in campaign_df.iterrows():
                seg = row[comp]
                gt = row['gt_image']
                composite_key = row['composite_key']

                # Load images
                seg_img = tifffile.imread(seg)
                gt_img = tifffile.imread(gt)

                # Calculate Jaccard scores for this image
                jaccard_scores = calculate_jaccard_scores(gt_img, seg_img)
                all_results[comp][campaign][composite_key] = jaccard_scores

                # Update set of all labels seen
                all_labels.update(jaccard_scores.keys())

                # Calculate average Jaccard for this image
                if jaccard_scores:
                    per_image_averages[comp][campaign][composite_key] = sum(jaccard_scores.values()) / len(
                        jaccard_scores)
                    campaign_label_scores.extend(list(jaccard_scores.values()))
                else:
                    per_image_averages[comp][campaign][composite_key] = 0.0

            # Calculate campaign average for this competitor
            if campaign_label_scores:
                per_campaign_averages[comp][campaign] = sum(campaign_label_scores) / len(campaign_label_scores)
                all_label_scores.extend(campaign_label_scores)
            else:
                per_campaign_averages[comp][campaign] = 0.0

        # Calculate overall average for this competitor
        if all_label_scores:
            overall_averages[comp] = sum(all_label_scores) / len(all_label_scores)
        else:
            overall_averages[comp] = 0.0

    # Format and display results
    print_results(all_results, per_image_averages, per_campaign_averages, overall_averages, campaigns)


@click.group()
def cli():
    pass

cli.add_command(evaluate_competitor)

if __name__ == '__main__':
    cli()