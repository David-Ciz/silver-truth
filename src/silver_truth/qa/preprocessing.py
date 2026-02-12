from typing import Optional, Any

import pandas as pd
import tifffile
import numpy as np
from pathlib import Path
from scipy.ndimage import find_objects, center_of_mass
from tqdm import tqdm
import logging

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
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
    centering: str = "competitor",
    exclude_competitors: Optional[list[Any]] = None,
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
        centering: Strategy for centering crops ('competitor', 'gt_mask', 'tracking_marker').
        exclude_competitors: List of competitor names to exclude from the dataset.
    """
    logging.info("Starting QA dataset creation.")
    logging.info(f"  - Dataset dataframe path: {dataset_dataframe_path}")
    logging.info(f"  - Output directory: {output_dir}")
    logging.info(f"  - Output parquet path: {output_parquet_path}")
    logging.info(f"  - Crop: {crop}")
    if crop:
        logging.info(f"  - Crop size: {crop_size}")
        logging.info(f"  - Centering: {centering}")

    df = load_dataframe_from_parquet_with_metadata(dataset_dataframe_path)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_path}")

    # Find project root (directory containing 'data' folder) for relative path conversion
    project_root = None
    for parent in output_path.parents:
        if (parent / "data").exists():
            project_root = parent
            break

    def to_relative_path(abs_path):
        """Convert absolute path to relative path starting from project root."""
        if abs_path is None:
            return None
        p = Path(abs_path)
        if project_root and p.is_absolute():
            try:
                return str(p.relative_to(project_root))
            except ValueError:
                return str(p)
        return str(p)

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
            "source_image",
            "tracking_marker",  # Typo in some datasets
        ]
        competitor_columns = [col for col in df.columns if col not in excluded_columns]
        logging.info(f"Inferred competitor columns: {competitor_columns}")

    if exclude_competitors:
        competitor_columns = [
            c for c in competitor_columns if c not in exclude_competitors
        ]
        logging.info(f"Columns after exclusion: {competitor_columns}")

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

        # Helper to get crop center
        def get_crop_center(label, segmentation, gt_image, row_data):
            if centering == "gt_mask":
                # Center on GT mask
                try:
                    gt_labeled = (gt_image == label).astype(int)
                    cy, cx = center_of_mass(gt_labeled)
                    return int(cy), int(cx)
                except Exception:
                    # Fallback to bounding box center if center_of_mass fails
                    try:
                        slice_y, slice_x = find_objects(gt_labeled)[0]
                        return (slice_y.start + slice_y.stop) // 2, (
                            slice_x.start + slice_x.stop
                        ) // 2
                    except IndexError:
                        logging.warning(
                            f"Could not find GT object for label {label}. Skipping."
                        )
                        return None

            elif centering == "tracking_marker":
                # Center on Tracking Marker
                marker_path_str = row_data.get("tracking_markers")
                if (
                    not marker_path_str
                    or pd.isna(marker_path_str)
                    or not Path(marker_path_str).exists()
                ):
                    # Fallback to GT mask if marker missing
                    logging.warning(
                        f"Tracking marker missing for {row_data['composite_key']}, using GT mask center."
                    )
                    return get_crop_center(
                        label, segmentation, gt_image, row_data
                    )  # Recursive call with GT fallback - potential infinite loop if logic wrong, but simplified here:
                    # Actually better to just inline the fallback logic to avoid recursion issues or change centering var.
                    # Let's just duplicate the GT logic for robustness or use a flag.

                try:
                    # Only read if we haven't already (optimization opportunity, but fine for now)
                    tra_image = tifffile.imread(marker_path_str)
                    tra_labeled = (tra_image == label).astype(int)
                    cy, cx = center_of_mass(tra_labeled)

                    if np.isnan(cy) or np.isnan(cx):
                        # If marker for this specific label is missing?
                        # TRA usually has markers for all cells.
                        # Check if label exists in TRA
                        if label not in tra_image:
                            logging.warning(
                                f"Label {label} not found in tracking marker image. Using GT center."
                            )
                            # FALLBACK TO GT
                            gt_labeled = (gt_image == label).astype(int)
                            cy, cx = center_of_mass(gt_labeled)
                            return int(cy), int(cx)

                    return int(cy), int(cx)
                except Exception as e:
                    logging.warning(
                        f"Error reading/processing tracking marker: {e}. Using GT center."
                    )
                    # FALLBACK TO GT
                    try:
                        gt_labeled = (gt_image == label).astype(int)
                        cy, cx = center_of_mass(gt_labeled)
                        return int(cy), int(cx)
                    except (ValueError, TypeError):
                        return None

            else:  # "competitor" (default)
                try:
                    labeled_segmentation = (segmentation == label).astype(int)
                    slice_y, slice_x = find_objects(labeled_segmentation)[0]
                    return (slice_y.start + slice_y.stop) // 2, (
                        slice_x.start + slice_x.stop
                    ) // 2
                except IndexError:
                    return None

        # Optimization: Pre-load/Pre-calculate shared centers if using a shared reference frame
        # If centering is NOT 'competitor', the center depends only on (GT or TRA) + Label, not the competitor.
        # We can calculate it once per label.
        shared_centers = {}
        if centering != "competitor":
            try:
                gt_image = tifffile.imread(gt_image_path)
                gt_labels = np.unique(gt_image)[1:]

                # We need to process labels that are in GT.
                # Note: The main loop iterates competitors, then finds intersection(seg, gt).
                # So we will only ever process labels present in GT. Good.
                pass
            except Exception as e:
                logging.error(f"Error pre-loading GT for centering: {e}")
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

                segmentation_labels = np.unique(segmentation)[1:]  # Exclude background
                gt_labels = np.unique(gt_image)[1:]

                # Process only the labels that are present in both the segmentation and the ground truth
                labels_to_process = np.intersect1d(
                    segmentation_labels, gt_labels, assume_unique=True
                )

                # Log a warning for labels present in the segmentation but not in the ground truth
                extra_labels = np.setdiff1d(
                    segmentation_labels, gt_labels, assume_unique=True
                )
                if extra_labels.size > 0:
                    logging.warning(
                        f"Segmentation {segmentation_path.name} contains labels not in ground truth: {extra_labels.tolist()}. These will be ignored."
                    )

                for label in labels_to_process:
                    if crop:
                        # Determine Center
                        if centering == "competitor":
                            # Use this specific segmentation's center
                            center_coords = get_crop_center(
                                label, segmentation, gt_image, row
                            )
                        else:
                            # Use shared center (GT or TRA)
                            if label not in shared_centers:
                                shared_centers[label] = get_crop_center(
                                    label, None, gt_image, row
                                )
                            center_coords = shared_centers[label]

                        if center_coords is None:
                            logging.warning(
                                f"Could not determine center for label {label} in {segmentation_path.name} with strategy {centering}. Skipping."
                            )
                            continue

                        center_y, center_x = center_coords
                        half_size = crop_size // 2

                        y_start, y_end = center_y - half_size, center_y + half_size
                        x_start, x_end = center_x - half_size, center_x + half_size

                        # 1. Calculate how much "out of bounds" we are on every side
                        pad_top = max(0, -y_start)
                        pad_bottom = max(0, y_end - raw_image.shape[0])
                        pad_left = max(0, -x_start)
                        pad_right = max(0, x_end - raw_image.shape[1])

                        # 2. Define the valid slice coordinates within the image
                        img_y_start = max(0, y_start)
                        img_y_end = min(raw_image.shape[0], y_end)
                        img_x_start = max(0, x_start)
                        img_x_end = min(raw_image.shape[1], x_end)

                        # 3. Crop the valid part of the image
                        raw_crop = raw_image[
                            img_y_start:img_y_end, img_x_start:img_x_end
                        ]
                        seg_crop = (
                            segmentation[img_y_start:img_y_end, img_x_start:img_x_end]
                            == label
                        ).astype(np.uint8) * 255

                        # 4. Pad symmetrically using the calculated offsets
                        # format: ((top, bottom), (left, right))
                        raw_crop = np.pad(
                            raw_crop,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode="constant",
                        )
                        seg_crop = np.pad(
                            seg_crop,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode="constant",
                        )

                        # Stack [Raw, Seg, GT] if aligning (or always? Let's add GT for reference if not competitor centering)
                        # Actually, for fusion we need the GT crop saved somewhere to evaluate against.
                        # The current format is 2 channels: [Raw, Seg].
                        # If we change it, it might break existing tools.
                        # But for the fusion experiment, we absolutely need the GT crop.

                        # Let's save a separate "GT" crop for each cell if it doesn't exist?
                        # Or just include it as a 3rd channel?
                        # Existing tools like `stacked_jaccard_logic` read channels 0 and 1.
                        # Adding a 3rd channel shouldn't break reading if they just read [0] and [1].

                        # Extract GT crop using same coordinates
                        gt_crop = (
                            gt_image[img_y_start:img_y_end, img_x_start:img_x_end]
                            == label
                        ).astype(np.uint8) * 255
                        gt_crop = np.pad(
                            gt_crop,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode="constant",
                        )

                        # We will save as 3 channels: Raw, Seg, GT_Mask
                        stacked_crop = np.stack([raw_crop, seg_crop, gt_crop], axis=0)

                        # Verify
                        assert stacked_crop.shape == (
                            3,
                            crop_size,
                            crop_size,
                        ), f"Shape mismatch: {stacked_crop.shape}"

                        # Save the stacked image
                        # Include campaign number in cell_id to distinguish between campaigns
                        cell_id = f"c{campaign_number}_{raw_image_path.stem}_{competitor}_{label}"
                        stacked_path = output_path / f"{cell_id}.tif"
                        tifffile.imwrite(stacked_path, stacked_crop)

                        # Collect metadata including crop coordinates
                        data_list.append(
                            {
                                "cell_id": cell_id,
                                "stacked_path": to_relative_path(stacked_path),
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
                                "centering": centering,
                                "original_image_path": to_relative_path(raw_image_path),
                                "gt_image": to_relative_path(row["gt_image"]),
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
                                "stacked_path": to_relative_path(stacked_path),
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
                                "original_image_path": to_relative_path(raw_image_path),
                                "gt_image": to_relative_path(row["gt_image"]),
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
