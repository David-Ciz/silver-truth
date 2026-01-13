import os
import subprocess
import logging
from enum import Enum
from typing import Optional
from src.ensemble.datasets import EnsembleDatasetC1
from src.ensemble.ensemble import _get_eval_sets
import tifffile
from tqdm import tqdm
import src.ensemble.external as ext
import segmentation_models_pytorch as smp
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.ndimage import find_objects

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def build_results_databank(
    original_parquet, results_path, output_parquet_path, crop_size=64
):
    # load original dataset
    df = ext.load_parquet(original_parquet)

    data_list = []
    competitor_columns = df.attrs.get("competitor_columns", [])

    # If no competitor_columns in attrs, infer them from columns
    if not competitor_columns:
        # Exclude standard columns to get competitor columns
        excluded_columns = [
            "composite_key",
            "campaign_number",
            "gt_image",
            "tracking_markers",
        ]
        competitor_columns = [col for col in df.columns if col not in excluded_columns]
        logging.info(f"Inferred competitor columns: {competitor_columns}")

    logging.info(f"Using competitor columns: {competitor_columns}")

    # Filter for rows that have ground truth images
    initial_count = len(df)

    # First filter: check if gt_image column has non-null values
    df_with_gt = df[df["gt_image"].notna()].copy()

    # Second filter: check if GT files actually exist on disk
    gt_exists_mask = df_with_gt["gt_image"].apply(
        lambda gt_path: Path(gt_path).exists() if gt_path else False
    )
    df_filtered = df_with_gt[gt_exists_mask].copy()

    filtered_count = len(df_filtered)
    logging.info(
        f"Filtered dataset: {initial_count} -> {filtered_count} images (only those with existing Ground Truth)"
    )

    if filtered_count == 0:
        logging.warning("No images with valid Ground Truth found. Exiting.")
        return

    # Use tqdm to create a progress bar for the main loop
    for __, row in tqdm(
        df_filtered.iterrows(), total=df_filtered.shape[0], desc="Processing images"
    ):
        # Construct raw image path from composite_key
        composite_key = row["composite_key"]  # e.g., "01_0061.tif"
        campaign_number = row["campaign_number"]  # e.g., "01"

        # Extract time frame number from composite_key (e.g., "0061" from "01_0061.tif")
        time_frame = composite_key.split("_")[1].split(".")[0]  # "0061"

        # Construct the path to the raw image by deriving it from GT image path
        gt_image_path = Path(row["gt_image"])
        # GT path: .../synchronized_data/DIC-C2DH-HeLa/01_GT/SEG/man_seg002.tif

        base_path = Path(results_path)
        # Try different formats for raw image files
        time_frame_int = int(time_frame)
        fused_path = None
        for format_str in [
            f"fused_{time_frame_int:03d}.tif",
            f"fused_{time_frame_int:04d}.tif",
            f"fused_{time_frame}.tif",
        ]:
            candidate_path = base_path / campaign_number / format_str
            if candidate_path.exists():
                fused_path = candidate_path
                break

        if fused_path is None:
            logger.error(
                f"Fused image file not found for time frame {time_frame} in {base_path / campaign_number}"
            )
            continue

        try:
            fused_img = tifffile.imread(fused_path)
            gt_image = tifffile.imread(gt_image_path)
        except Exception as e:
            logging.error(
                f"Error reading fused image or ground truth file for {fused_path}: {e}"
            )
            continue

        labels = np.unique(fused_img)[1:]  # Exclude background
        ### guarantees that labels are synchronized
        gt_labels_lst = np.unique(gt_image)[1:].tolist()
        assert all([lbl in gt_labels_lst for lbl in labels.tolist()])
        ###

        for label in labels:
            # Get bounding box for the cell
            try:
                labeled_segmentation = (fused_img == label).astype(int)
                slice_y, slice_x = find_objects(labeled_segmentation)[0]
            except IndexError:
                logging.warning(
                    f"Could not find object for label {label} in {fused_path.name}. Skipping."
                )
                continue

            # Center and crop
            center_y, center_x = (
                (slice_y.start + slice_y.stop) // 2,
                (slice_x.start + slice_x.stop) // 2,
            )
            half_size = crop_size // 2

            y_start, y_end = center_y - half_size, center_y + half_size
            x_start, x_end = center_x - half_size, center_x + half_size

            # Ensure crops are within image bounds
            y_start, y_end = max(0, y_start), min(gt_image.shape[0], y_end)
            x_start, x_end = max(0, x_start), min(gt_image.shape[1], x_end)

            gt_crop = gt_image[y_start:y_end, x_start:x_end]
            fused_crop = (fused_img[y_start:y_end, x_start:x_end] == label).astype(
                np.uint8
            ) * 255

            # Pad if crop is smaller than crop_size
            pad_y = crop_size - gt_image.shape[0]
            pad_x = crop_size - gt_image.shape[1]

            if pad_y > 0 or pad_x > 0:
                gt_crop = np.pad(gt_crop, ((0, pad_y), (0, pad_x)), "constant")
                fused_crop = np.pad(fused_crop, ((0, pad_y), (0, pad_x)), "constant")

            gt_crop = (gt_crop == label).astype(np.uint8) * 255
            # contains the rest of the gt segmentations, shown in blue
            # blue_layer = np.logical_and(gt_crop > 0, gt_crop != row.label).astype(np.uint8) * 255
            blue_layer = np.zeros(
                (crop_size, crop_size), dtype=np.uint8
            )  # empty blue layer

            # Stack the crops
            stacked_crop = np.stack([fused_crop, gt_crop, blue_layer], axis=0)

            # Save the stacked image
            # Include campaign number in cell_id to distinguish between campaigns
            cell_id = f"c{campaign_number}_t{time_frame}_{label}"
            new_image_path = base_path / f"{cell_id}.tif"
            tifffile.imwrite(new_image_path, stacked_crop)

            # Collect metadata including crop coordinates
            data_list.append(
                {
                    "campaign": campaign_number,
                    "image_id": time_frame,
                    "label": label,
                    "crop_size": crop_size,
                    "image_path": str(new_image_path),
                }
            )

    output_df = pd.DataFrame(data_list)
    output_df.to_parquet(output_parquet_path)
    # compress images
    ext.compress_images(results_path, recursive=False)


def generate_evaluation(parquet_path: str, split_type: str = "test") -> str:
    """
    Generate a parquet file with the evaluation of the fusion results.
    """

    # load dataset
    # TODO: what if it's other dataset?
    dataset = EnsembleDatasetC1(parquet_path, split_type)
    result_set, target_set = _get_eval_sets(dataset)

    # calculate metrics
    tp, fp, fn, tn = smp.metrics.get_stats(
        result_set.long(), target_set.long(), mode="binary", threshold=0.5
    )  # type: ignore
    iou = smp.metrics.iou_score(tp, fp, fn, tn)
    f1 = smp.metrics.f1_score(tp, fp, fn, tn)

    # create dataframe
    data_list = []
    # load dataframe
    df = ext.load_parquet(parquet_path)
    df = df[df["split"] == split_type]

    for index, row in enumerate(df.itertuples()):
        data_list.append(
            {
                "image_path": row.image_path,
                "tp": tp[index].item(),
                "fp": fp[index].item(),
                "fn": fn[index].item(),
                "tn": tn[index].item(),
                "iou": iou[index].item(),
                "f1": f1[index].item(),
            }
        )

    # save results
    output_prefix = parquet_path.split("/")[-1]
    base_path = parquet_path[: -len(output_prefix)]
    output_name = output_prefix[: -len(".parquet")]
    output_parquet_path = f"{base_path}eval_{output_name}_{split_type}set.parquet"
    output_df = pd.DataFrame(data_list)
    output_df.to_parquet(output_parquet_path)
    return output_parquet_path
