import os
import subprocess
import logging
from enum import Enum
from typing import Optional
import pandas as pd
from pathlib import Path
import re

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FusionModel(Enum):
    """Enumeration for the fusion models available in Fusers.java."""

    THRESHOLD_FLAT = "Threshold - flat weights"
    THRESHOLD_USER = "Threshold - user weights"
    MAJORITY_FLAT = "Majority - flat weights"
    SIMPLE = "SIMPLE"
    BIC_FLAT_VOTING = "BICv2 with FlatVoting, SingleMaskFailSafe and CollisionResolver"
    BIC_WEIGHTED_VOTING = (
        "BICv2 with WeightedVoting, SingleMaskFailSafe and CollisionResolver"
    )


def fuse_segmentations(
    jar_path: str,
    job_file_path: str,
    output_path_pattern: str,
    time_points: str,
    num_threads: int,
    fusion_model: FusionModel,
    threshold: float = 1.0,
    cmv_mode: Optional[str] = None,
    seg_eval_folder: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Runs the Fusers Java tool by calling the flexible RunFusersCli wrapper."""
    # (The body of this function remains exactly the same as before)
    jar_path = os.path.abspath(jar_path)
    job_file_path = os.path.abspath(job_file_path)
    output_dir = os.path.dirname(os.path.abspath(output_path_pattern))
    if not os.path.isdir(output_dir):
        logging.info(f"Output directory '{output_dir}' does not exist. Creating it...")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Successfully created output directory: {output_dir}")
    if seg_eval_folder:
        seg_eval_folder = os.path.abspath(seg_eval_folder)

    logging.info(f"Preparing to run Fusers with model: '{fusion_model.value}'")
    logging.info(f"Job file: {job_file_path}")

    command = [
        "java",
        "-cp",
        jar_path,
        "de.mpicbg.ulman.fusion.RunFusersCli",
        fusion_model.value,
        job_file_path,
        str(threshold),
        output_path_pattern,
        time_points,
        str(num_threads),
    ]
    if cmv_mode:
        command.append(cmv_mode)
    if seg_eval_folder:
        command.append(seg_eval_folder)

    logging.info(f"Executing command: {' '.join(command)}")
    stdout_pipe = None if debug else subprocess.PIPE
    stderr_pipe = None if debug else subprocess.PIPE
    result = subprocess.run(
        command, stdout=stdout_pipe, stderr=stderr_pipe, text=True, check=False
    )

    if result.returncode != 0:
        logging.error("The Fusers Java process failed.")
        if not debug:
            logging.error(f"Return Code: {result.returncode}")
            logging.error(f"STDOUT:\n{result.stdout}")
            logging.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError("Execution of Fusers (RunFusersCli) Java tool failed.")
    else:
        logging.info("Fusers Java process completed successfully.")


def extract_timepoint_from_filename(filename):
    """
    Extract timepoint number from fused image filename.
    Examples:
        - mask_0000.tif -> 0
        - mask_0042.tif -> 42
        - fused_0123.tif -> 123
    """
    if pd.isna(filename) or not filename:
        return None
    
    # Match patterns like mask_0000.tif, fused_0123.tif, etc.
    match = re.search(r'_(\d+)\.tif$', filename)
    if match:
        return int(match.group(1))
    return None


def add_fused_images_to_dataframe_logic(dataset_name, base_dir, parquet_path=None, fused_dir=None, output_path=None,
                                       fusion_model="threshold_flat", fusion_threshold=1.0, 
                                       fusion_timepoints_range="0-61"):
    """
    Process a single dataset to add fused image paths and fusion metadata.

    Args:
        dataset_name: Name of the dataset (e.g., 'BF-C2DL-MuSC')
        base_dir: Base directory path
        parquet_path: Optional custom path to parquet file
        fused_dir: Optional custom path to fused images directory
        output_path: Optional custom output path for parquet with fused images
        fusion_model: Fusion model used (default: 'threshold_flat')
        fusion_threshold: Fusion threshold used (default: 1.0)
        fusion_timepoints_range: Timepoints range used in fusion (default: '0-61')
    """
    # Construct paths - use custom paths if provided, otherwise use default structure
    if parquet_path is None:
        parquet_path = (
            Path(base_dir) / "dataframes" / f"{dataset_name}_dataset_dataframe.parquet"
        )
    else:
        parquet_path = Path(parquet_path)
    
    if fused_dir is None:
        fused_images_dir = Path(base_dir) / "fused_results"
    else:
        fused_images_dir = Path(fused_dir)
    
    if output_path is None:
        output_dir = Path(base_dir) / "fused_results_parquet"
        output_parquet_path = (
            output_dir / f"{dataset_name}_dataset_dataframe_with_fused.parquet"
        )
    else:
        output_parquet_path = Path(output_path)
        output_dir = output_parquet_path.parent

    # Check if parquet file exists
    if not parquet_path.exists():
        print(f"Warning: Parquet file not found: {parquet_path}")
        return False

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataframe
    print(f"Processing dataset: {dataset_name}")
    df = pd.read_parquet(parquet_path)

    # Create new columns for fused images and metadata, initialized to None
    df["fused_images"] = None
    df["fusion_timepoint"] = None
    df["fusion_model"] = None
    df["fusion_threshold"] = None
    df["fusion_timepoints_range"] = None

    # Get unique campaign numbers from the dataset
    campaign_numbers = df["campaign_number"].unique()
    print(f"Found campaigns: {campaign_numbers}")

    # Process each campaign
    for campaign in campaign_numbers:
        # Build a dictionary of fused images for THIS campaign only
        campaign_fused_dir = fused_images_dir / campaign
        existing_fused_images = {}
        
        # If campaign subfolder exists, search there
        if campaign_fused_dir.exists():
            for pattern in ["*_fused_*.tif", "mask_*.tif", "fused_*.tif"]:
                for p in campaign_fused_dir.glob(pattern):
                    existing_fused_images[p.name] = str(p)
            print(f"Found {len(existing_fused_images)} fused images in {campaign_fused_dir}")
        else:
            # Fallback: search in root fused_images_dir for campaign-specific patterns
            for pattern in ["*_fused_*.tif", "mask_*.tif", "fused_*.tif"]:
                for p in fused_images_dir.rglob(pattern):
                    # Only include if path contains the campaign number
                    if f"/{campaign}/" in str(p).replace("\\", "/") or f"_{campaign}_" in p.name:
                        existing_fused_images[p.name] = str(p)
            print(f"Found {len(existing_fused_images)} fused images for campaign {campaign}")
        
        # Filter for current campaign rows and sort by composite_key to match fusion order
        df_campaign = df[df["campaign_number"] == campaign].copy()
        df_campaign = df_campaign.sort_values("composite_key").reset_index(drop=True)

        # Map fused images based on composite_key TTT/TTTT pattern
        fused_image_mapping = {}
        for index, row in df_campaign.iterrows():
            # Extract the time point number from composite_key (e.g., "01_0005.tif" -> "005")
            composite_key = row["composite_key"]
            if "_" in composite_key:
                time_part = composite_key.split("_")[1].replace(".tif", "")
                # Convert to integer and determine the correct format based on existing files
                time_num = int(time_part)

                # Try multiple filename patterns
                possible_filenames = [
                    f"{dataset_name}_{campaign}_fused_{time_num:03d}.tif",
                    f"{dataset_name}_{campaign}_fused_{time_num:04d}.tif",
                    f"mask_{time_num:03d}.tif",
                    f"mask_{time_num:04d}.tif",
                    f"fused_{time_num:03d}.tif",
                    f"fused_{time_num:04d}.tif",
                ]

                fused_path = None
                for filename in possible_filenames:
                    if filename in existing_fused_images:
                        fused_path = existing_fused_images[filename]
                        break
                
                # Debug: print first few misses for campaign 02
                if fused_path is None and campaign == "02" and index < 3:
                    print(f"  DEBUG: Could not find fused image for {composite_key}")
                    print(f"    Tried filenames: {possible_filenames[:4]}")
                    print(f"    Available (first 5): {list(existing_fused_images.keys())[:5]}")
                
                if fused_path:
                    fused_image_mapping[row["composite_key"]] = fused_path
            else:
                # Fallback to sequential index if composite_key format is unexpected
                possible_filenames = [
                    f"{dataset_name}_{campaign}_fused_{index:03d}.tif",
                    f"{dataset_name}_{campaign}_fused_{index:04d}.tif",
                    f"mask_{index:03d}.tif",
                    f"mask_{index:04d}.tif",
                ]
                for filename in possible_filenames:
                    if filename in existing_fused_images:
                        fused_image_mapping[row["composite_key"]] = existing_fused_images[filename]
                        break

        # Apply the mapping to the DataFrame
        # Use composite_key from the original df, not df_campaign, to avoid index mismatch
        campaign_mask = df["campaign_number"] == campaign
        df.loc[campaign_mask, "fused_images"] = df.loc[campaign_mask, "composite_key"].map(fused_image_mapping)
        
        # Add fusion metadata for rows with fused images
        has_fused = campaign_mask & df["fused_images"].notna()
        
        # Extract timepoint from fused image filename
        df.loc[has_fused, "fusion_timepoint"] = df.loc[has_fused, "fused_images"].apply(
            lambda x: extract_timepoint_from_filename(os.path.basename(x))
        )
        
        # Add fusion configuration metadata
        df.loc[has_fused, "fusion_model"] = fusion_model
        df.loc[has_fused, "fusion_threshold"] = fusion_threshold
        df.loc[has_fused, "fusion_timepoints_range"] = fusion_timepoints_range

        print(f"Campaign {campaign}: {len(fused_image_mapping)} fused images mapped")

    # Save the modified DataFrame
    df.to_parquet(output_parquet_path, index=False)

    print(f"Modified dataframe saved to: {output_parquet_path}")

    # Show summary for each campaign
    for campaign in campaign_numbers:
        campaign_data = df[df["campaign_number"] == campaign][
            ["composite_key", "fused_images", "fusion_timepoint", "fusion_model", 
             "fusion_threshold", "fusion_timepoints_range"]
        ]
        mapped_count = campaign_data["fused_images"].notna().sum()
        total_count = len(campaign_data)
        print(
            f"Campaign {campaign}: {mapped_count}/{total_count} images have fused counterparts"
        )
        if mapped_count > 0:
            print(campaign_data[campaign_data["fused_images"].notna()].head())
    
    # Show fusion metadata summary
    if df["fused_images"].notna().any():
        print("\nFusion metadata summary:")
        print(f"  - Fusion model: {df['fusion_model'].unique()}")
        print(f"  - Fusion threshold: {df['fusion_threshold'].unique()}")
        print(f"  - Timepoints range: {df['fusion_timepoints_range'].unique()}")
        print(f"  - Timepoint values: {df['fusion_timepoint'].min():.0f} - {df['fusion_timepoint'].max():.0f}")

    return True


def process_all_datasets_logic(base_dir):
    """Process all datasets in the dataframes directory."""
    dataframes_dir = Path(base_dir) / "dataframes"

    if not dataframes_dir.exists():
        print(f"Error: Dataframes directory not found: {dataframes_dir}")
        return

    # Find all parquet files
    parquet_files = list(dataframes_dir.glob("*_dataset_dataframe.parquet"))

    if not parquet_files:
        print("No dataset parquet files found!")
        return

    print(f"Found {len(parquet_files)} datasets to process:")

    success_count = 0
    for parquet_file in parquet_files:
        # Extract dataset name from filename
        dataset_name = parquet_file.name.replace("_dataset_dataframe.parquet", "")
        print(f"\n{'='*50}")

        if add_fused_images_to_dataframe_logic(dataset_name, base_dir):
            success_count += 1
        else:
            print(f"Failed to process: {dataset_name}")

    print(f"\n{'='*50}")
    print(
        f"Processing complete: {success_count}/{len(parquet_files)} datasets processed successfully"
    )
