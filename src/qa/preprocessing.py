import pandas as pd
import tifffile
import numpy as np
from pathlib import Path
from scipy.ndimage import find_objects
from tqdm import tqdm
import logging

from src.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)

# Basic Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_qa_dataset(
    dataset_dataframe_path: str,
    output_dir: str,
    output_parquet_path: str,
    crop: bool = True,
    crop_size: int = 64,
):
    """
    Generates a QA dataset from images that have corresponding Ground Truth annotations.
    If crop=True, it creates stacked cell crops.
    If crop=False, it creates stacked full images with individual cell masks.

    Only processes images that have valid Ground Truth files (gt_image column is not null
    and the GT file exists on disk).

    Args:
        dataset_dataframe_path: Path to the main Parquet file that tracks all the
            images and competitors.
        output_dir: The directory where the stacked images will be saved.
        output_parquet_path: The path to save the resulting dataframe.parquet file.
        crop: Whether to crop individual cells or use the full image.
        crop_size: If cropping, the size of the bounding box for the cell crops.
    """
    logging.info("Starting QA dataset creation.")
    logging.info(f"  - Dataset dataframe path: {dataset_dataframe_path}")
    logging.info(f"  - Output directory: {output_dir}")
    logging.info(f"  - Output parquet path: {output_parquet_path}")
    logging.info(f"  - Crop: {crop}")
    if crop:
        logging.info(f"  - Crop size: {crop_size}")

    df = load_dataframe_from_parquet_with_metadata(dataset_dataframe_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_path}")

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
    for index, row in tqdm(
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
        # Raw path should be: .../synchronized_data/DIC-C2DH-HeLa/01/t002.tif or t0002.tif
        gt_parts = gt_image_path.parts
        synchronized_data_idx = gt_parts.index("synchronized_data")
        _ = gt_parts[synchronized_data_idx + 1]  # e.g., "DIC-C2DH-HeLa"
        base_path = Path(
            *gt_parts[: synchronized_data_idx + 2]
        )  # e.g., "C:\Users\wei0068\Desktop\Cell_Tracking\synchronized_data\DIC-C2DH-HeLa"

        # Try different formats for raw image files
        time_frame_int = int(time_frame)
        raw_image_path = None
        for format_str in [
            f"t{time_frame_int:03d}.tif",
            f"t{time_frame_int:04d}.tif",
            f"t{time_frame}.tif",
        ]:
            candidate_path = base_path / campaign_number / format_str
            if candidate_path.exists():
                raw_image_path = candidate_path
                break

        if raw_image_path is None:
            logger.error(
                f"Raw image file not found for time frame {time_frame} in {base_path / campaign_number}"
            )
            continue

        for competitor in competitor_columns:
            segmentation_path_str = row[competitor]

            if segmentation_path_str and Path(segmentation_path_str).exists():
                segmentation_path = Path(segmentation_path_str)
                try:
                    raw_image = tifffile.imread(raw_image_path)
                    segmentation = tifffile.imread(segmentation_path)
                    gt_image = tifffile.imread(gt_image_path)
                except Exception as e:
                    logging.error(
                        f"Error reading image, segmentation or ground truth file for {raw_image_path.stem}: {e}"
                    )
                    continue

                labels = np.unique(segmentation)[1:]  # Exclude background
                ### guarantees that labels are synchronized
                gt_labels_lst = np.unique(gt_image)[1:].tolist()
                assert(all([lbl in gt_labels_lst for lbl in labels.tolist()]))
                ###

                for label in labels:
                    if crop:
                        # Get bounding box for the cell
                        try:
                            labeled_segmentation = (segmentation == label).astype(int)
                            slice_y, slice_x = find_objects(labeled_segmentation)[0]
                        except IndexError:
                            logging.warning(
                                f"Could not find object for label {label} in {segmentation_path.name}. Skipping."
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
                        y_start, y_end = max(0, y_start), min(raw_image.shape[0], y_end)
                        x_start, x_end = max(0, x_start), min(raw_image.shape[1], x_end)

                        raw_crop = raw_image[y_start:y_end, x_start:x_end]
                        seg_crop = (
                            segmentation[y_start:y_end, x_start:x_end] == label
                        ).astype(np.uint8) * 255

                        # Pad if crop is smaller than crop_size
                        pad_y = crop_size - raw_crop.shape[0]
                        pad_x = crop_size - raw_crop.shape[1]

                        if pad_y > 0 or pad_x > 0:
                            raw_crop = np.pad(
                                raw_crop, ((0, pad_y), (0, pad_x)), "constant"
                            )
                            seg_crop = np.pad(
                                seg_crop, ((0, pad_y), (0, pad_x)), "constant"
                            )

                        # Stack the crops
                        stacked_crop = np.stack([raw_crop, seg_crop], axis=0)

                        # Save the stacked image
                        # Include campaign number in cell_id to distinguish between campaigns
                        cell_id = f"c{campaign_number}_{raw_image_path.stem}_{competitor}_{label}"
                        stacked_path = output_path / f"{cell_id}.tif"
                        tifffile.imwrite(stacked_path, stacked_crop)

                        # Collect metadata including crop coordinates
                        data_list.append(
                            {
                                "cell_id": cell_id,
                                "stacked_path": str(stacked_path),
                                "original_image_key": raw_image_path.stem,
                                "campaign_number": campaign_number,
                                "competitor": competitor,
                                "label": label,
                                "crop_y_start": y_start,
                                "crop_y_end": y_end,
                                "crop_x_start": x_start,
                                "crop_x_end": x_end,
                                "original_center_y": center_y,
                                "original_center_x": center_x,
                                "crop_size": crop_size,
                                "gt_image": row["gt_image"],
                            }
                        )
                    else:  # not cropping
                        # Create a mask for the individual cell
                        cell_mask = (segmentation == label).astype(np.uint8) * 255

                        # Stack the raw image and the cell mask
                        stacked_image = np.stack([raw_image, cell_mask], axis=0)

                        # Save the stacked image
                        # Include campaign number in cell_id to distinguish between campaigns
                        cell_id = f"c{campaign_number}_{raw_image_path.stem}_{competitor}_{label}"
                        stacked_path = output_path / f"{cell_id}.tif"
                        tifffile.imwrite(stacked_path, stacked_image)

                        # Collect metadata (no cropping, so full image dimensions)
                        data_list.append(
                            {
                                "cell_id": cell_id,
                                "stacked_path": str(stacked_path),
                                "original_image_key": raw_image_path.stem,
                                "campaign_number": campaign_number,
                                "competitor": competitor,
                                "label": label,
                                "crop_y_start": 0,
                                "crop_y_end": raw_image.shape[0],
                                "crop_x_start": 0,
                                "crop_x_end": raw_image.shape[1],
                                "original_center_y": raw_image.shape[0] // 2,
                                "original_center_x": raw_image.shape[1] // 2,
                                "crop_size": None,  # No cropping applied
                                "gt_image": row["gt_image"],
                            }
                        )
            else:
                logging.warning(
                    f"Segmentation file not found for {raw_image_path.stem} and competitor {competitor}. Skipping."
                )

    output_df = pd.DataFrame(data_list)
    output_df.to_parquet(output_parquet_path)

    logging.info("QA dataset creation complete.")
    if crop:
        logging.info(
            f"  - {len(output_df)} cropped cell images created in {output_path}"
        )
    else:
        logging.info(
            f"  - {len(output_df)} full-size images with cell masks created in {output_path}"
        )
    logging.info(f"  - QA dataframe saved to {output_parquet_path}")
