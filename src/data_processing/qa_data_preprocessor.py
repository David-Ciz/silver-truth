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


def create_qa_dataset(
    dataset_dataframe_path: str,
    output_dir: str,
    output_parquet_path: str,
    crop: bool = True,
    crop_size: int = 64,
):
    """
    Generates a QA dataset.
    If crop=True, it creates stacked cell crops.
    If crop=False, it creates stacked full images with individual cell masks.

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

    # Use tqdm to create a progress bar for the main loop
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
        raw_image_path = Path(row["source_image"])

        for competitor in competitor_columns:
            segmentation_path_str = row[competitor]

            if segmentation_path_str and Path(segmentation_path_str).exists():
                segmentation_path = Path(segmentation_path_str)
                try:
                    raw_image = tifffile.imread(raw_image_path)
                    segmentation = tifffile.imread(segmentation_path)
                except Exception as e:
                    logging.error(
                        f"Error reading image or segmentation file for {raw_image_path.stem}: {e}"
                    )
                    continue

                labels = np.unique(segmentation)[1:]  # Exclude background

                for label in labels:
                    if crop:
                        # Get bounding box for the cell
                        try:
                            slice_y, slice_x = find_objects(segmentation == label)[0]
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
                        cell_id = f"{raw_image_path.stem}_{competitor}_{label}"
                        stacked_path = output_path / f"{cell_id}.tif"
                        tifffile.imwrite(stacked_path, stacked_crop)

                        # Collect metadata
                        data_list.append(
                            {
                                "cell_id": cell_id,
                                "stacked_path": str(stacked_path),
                                "original_image_key": raw_image_path.stem,
                                "competitor": competitor,
                                "label": label,
                            }
                        )
                    else:  # not cropping
                        # Create a mask for the individual cell
                        cell_mask = (segmentation == label).astype(np.uint8) * 255

                        # Stack the raw image and the cell mask
                        stacked_image = np.stack([raw_image, cell_mask], axis=0)

                        # Save the stacked image
                        cell_id = f"{raw_image_path.stem}_{competitor}_{label}"
                        stacked_path = output_path / f"{cell_id}.tif"
                        tifffile.imwrite(stacked_path, stacked_image)

                        # Collect metadata
                        data_list.append(
                            {
                                "cell_id": cell_id,
                                "stacked_path": str(stacked_path),
                                "original_image_key": raw_image_path.stem,
                                "competitor": competitor,
                                "label": label,
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
