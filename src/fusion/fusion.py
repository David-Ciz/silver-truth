import os
import subprocess
import logging
from enum import Enum
from typing import Optional
import pandas as pd
from pathlib import Path

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


def add_fused_images_to_dataframe_logic(dataset_name, base_dir):
    """
    Process a single dataset to add fused image paths.

    Args:
        dataset_name: Name of the dataset (e.g., 'Fluo-C3DH-A549')
        base_dir: Base directory path
    """
    # Construct paths
    parquet_path = (
        Path(base_dir) / "dataframes" / f"{dataset_name}_dataset_dataframe.parquet"
    )
    fused_images_dir = Path(base_dir) / "fused_results"
    output_dir = Path(base_dir) / "fused_results_parquet"
    output_parquet_path = (
        output_dir / f"{dataset_name}_dataset_dataframe_with_fused.parquet"
    )

    # Check if parquet file exists
    if not parquet_path.exists():
        print(f"Warning: Parquet file not found: {parquet_path}")
        return False

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataframe
    print(f"Processing dataset: {dataset_name}")
    df = pd.read_parquet(parquet_path)

    # Create a new column for fused images, initialized to None
    df["fused_images"] = None

    # Get a set of existing fused image filenames for efficient lookup
    existing_fused_images = {
        p.name for p in Path(fused_images_dir).glob("*_fused_*.tif")
    }

    # Get unique campaign numbers from the dataset
    campaign_numbers = df["campaign_number"].unique()
    print(f"Found campaigns: {campaign_numbers}")

    # Process each campaign
    for campaign in campaign_numbers:
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

                # Try TTT format first (3 digits) as it's more common
                fused_filename_ttt = (
                    f"{dataset_name}_{campaign}_fused_{time_num:03d}.tif"
                )
                # Also try TTTT format (4 digits) as fallback
                fused_filename_tttt = (
                    f"{dataset_name}_{campaign}_fused_{time_num:04d}.tif"
                )

                if fused_filename_ttt in existing_fused_images:
                    fused_filename = fused_filename_ttt
                elif fused_filename_tttt in existing_fused_images:
                    fused_filename = fused_filename_tttt
                else:
                    continue  # No matching fused file found
            else:
                # Fallback to sequential index if composite_key format is unexpected
                fused_filename = f"{dataset_name}_{campaign}_fused_{index:03d}.tif"
                if fused_filename not in existing_fused_images:
                    fused_filename = f"{dataset_name}_{campaign}_fused_{index:04d}.tif"
                if fused_filename not in existing_fused_images:
                    continue

            if fused_filename in existing_fused_images:
                fused_image_mapping[row["composite_key"]] = str(
                    Path(fused_images_dir) / fused_filename
                )

        # Apply the mapping to the DataFrame
        df.loc[df["campaign_number"] == campaign, "fused_images"] = df_campaign[
            "composite_key"
        ].map(fused_image_mapping)

        print(f"Campaign {campaign}: {len(fused_image_mapping)} fused images mapped")

    # Save the modified DataFrame
    df.to_parquet(output_parquet_path, index=False)

    print(f"Modified dataframe saved to: {output_parquet_path}")

    # Show summary for each campaign
    for campaign in campaign_numbers:
        campaign_data = df[df["campaign_number"] == campaign][
            ["composite_key", "fused_images"]
        ]
        mapped_count = campaign_data["fused_images"].notna().sum()
        total_count = len(campaign_data)
        print(
            f"Campaign {campaign}: {mapped_count}/{total_count} images have fused counterparts"
        )
        if mapped_count > 0:
            print(campaign_data[campaign_data["fused_images"].notna()].head())

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
