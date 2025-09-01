import os
import re
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
    BIC_WEIGHTED_VOTING = "BICv2 with WeightedVoting, SingleMaskFailSafe and CollisionResolver"


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


def add_fused_images_to_dataframe_logic(
    input_parquet_path: Path,
    output_parquet_path: Path,
    fused_images_dir: Path,
    dataset_name: str,
) -> bool:
    """
    Process a single dataset to add fused image paths.
    Args:
        input_parquet_path: Path to the input parquet file.
        output_parquet_path: Path to save the updated parquet file.
        fused_images_dir: Directory containing the fused images.
        dataset_name: Name of the dataset (e.g., 'Fluo-C3DH-A549')
    """
    if not input_parquet_path.exists():
        print(f"Warning: Parquet file not found: {input_parquet_path}")
        return False

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing dataset: {dataset_name}")
    df = pd.read_parquet(input_parquet_path)

    df["fused_images"] = None

    existing_fused_images = {p.name for p in fused_images_dir.glob("*_fused_*.tif")}

    campaign_numbers = df["campaign_number"].unique()
    print(f"Found campaigns: {campaign_numbers}")

    df_sorted = df.sort_values("composite_key").reset_index(drop=True)

    for campaign in campaign_numbers:
        df_campaign = df_sorted[df_sorted["campaign_number"] == campaign].copy()

        fused_image_mapping = {}
        for index, row in df_campaign.iterrows():
            composite_key = row["composite_key"]
            match = re.search(r"(\d+)\.tif", str(composite_key))
            if match:
                time_num = int(match.group(1))
                fused_filename_ttt = f"{dataset_name}_{campaign}_fused_{time_num:03d}.tif"
                fused_filename_tttt = f"{dataset_name}_{campaign}_fused_{time_num:04d}.tif"

                if fused_filename_ttt in existing_fused_images:
                    fused_filename = fused_filename_ttt
                elif fused_filename_tttt in existing_fused_images:
                    fused_filename = fused_filename_tttt
                else:
                    continue
                
                fused_image_mapping[composite_key] = str(
                    fused_images_dir / fused_filename
                )

        df.loc[df["campaign_number"] == campaign, "fused_images"] = df[
            "composite_key"
        ].map(fused_image_mapping)

        print(f"Campaign {campaign}: {len(fused_image_mapping)} fused images mapped")

    df.to_parquet(output_parquet_path, index=False)

    print(f"Modified dataframe saved to: {output_parquet_path}")

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

    parquet_files = list(dataframes_dir.glob("*_dataset_dataframe.parquet"))

    if not parquet_files:
        print("No dataset parquet files found!")
        return

    print(f"Found {len(parquet_files)} datasets to process:")

    success_count = 0
    for parquet_file in parquet_files:
        dataset_name = parquet_file.name.replace("_dataset_dataframe.parquet", "")
        print(f"\n{'='*50}")

        output_dir = Path(base_dir) / "fused_results_parquet"
        output_parquet_path = output_dir / f"{dataset_name}_dataset_dataframe_with_fused.parquet"
        fused_images_dir = Path(base_dir) / "fused_results"

        if add_fused_images_to_dataframe_logic(
            input_parquet_path=parquet_file,
            output_parquet_path=output_parquet_path,
            fused_images_dir=fused_images_dir,
            dataset_name=dataset_name,
        ):
            success_count += 1
        else:
            print(f"Failed to process: {dataset_name}")

    print(f"\n{'='*50}")
    print(
        f"Processing complete: {success_count}/{len(parquet_files)} datasets processed successfully"
    )
