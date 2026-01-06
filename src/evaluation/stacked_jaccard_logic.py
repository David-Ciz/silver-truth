import pandas as pd
import tifffile
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm


def jaccard_score(mask1, mask2):
    """
    Calculates the Jaccard score (IoU) between two binary masks.

    Args:
        mask1: First binary mask (numpy array)
        mask2: Second binary mask (numpy array)

    Returns:
        Jaccard score (float between 0 and 1)
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    if union_sum == 0:
        return 1.0 if intersection_sum == 0 else 0.0

    return intersection_sum / union_sum


def f1_score(mask1, mask2):
    """
    Calculates the F1 score (Dice coefficient) between two binary masks.

    Args:
        mask1: First binary mask (numpy array)
        mask2: Second binary mask (numpy array)

    Returns:
        F1 score (float between 0 and 1)
    """
    tp = np.sum(np.logical_and(mask1, mask2))
    fp = np.sum(np.logical_and(np.logical_not(mask1), mask2))
    fn = np.sum(np.logical_and(mask1, np.logical_not(mask2)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def calculate_evaluation_metrics(parquet_path: Path):
    """
    Calculates Jaccard and F1 scores for FULL-SIZE (non-cropped) QA images.

    This function is designed for parquet files where crop_size=None (full images).
    It compares the segmentation mask (layer 1 of stacked image) with the
    ground truth mask for a specific cell label.

    Stacked image structure:
        - Layer 0: Raw microscope image (source image)
        - Layer 1: Segmentation mask (competitor's segmentation, binary 0/255)

    The GT mask is created by reading the full GT image and extracting pixels
    matching the cell label.

    NOTE: For cropped QA images (crop_size != None), use
          calculate_evaluation_metrics_cropped() instead.

    Args:
        parquet_path: Path to the input parquet file. Results are saved back to this file.
    """
    logging.info(f"Loading parquet file: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logging.error(f"Failed to load parquet file: {e}")
        return

    if "stacked_path" not in df.columns:
        logging.error("Parquet file must contain a 'stacked_path' column.")
        return

    jaccard_scores = []
    f1_scores = []

    for index, row in tqdm(
        df.iterrows(), total=df.shape[0], desc="Calculating Evaluation Metrics"
    ):
        stacked_path = row["stacked_path"]
        gt_path = row["gt_image"]
        label = row["label"]
        try:
            image_stack = tifffile.imread(stacked_path)
            if image_stack.shape[0] < 2:
                logging.warning(
                    f"Image stack has less than 2 layers, skipping: {stacked_path}"
                )
                jaccard_scores.append(np.nan)
                f1_scores.append(np.nan)
                continue

            # Layer 1 is the segmentation mask (binary: 0 or 255)
            seg_mask = image_stack[1]

            # Read full GT image and create binary mask for specific label
            gt_image = tifffile.imread(gt_path)
            gt_mask = (gt_image == label).astype(np.uint8)

            # Calculate metrics
            j_score = jaccard_score(gt_mask, seg_mask)
            f1 = f1_score(gt_mask, seg_mask)
            jaccard_scores.append(j_score)
            f1_scores.append(f1)

        except FileNotFoundError:
            logging.warning(f"File not found, skipping: {stacked_path}")
            jaccard_scores.append(np.nan)
            f1_scores.append(np.nan)
        except Exception as e:
            logging.error(f"Error processing {stacked_path}: {e}")
            jaccard_scores.append(np.nan)
            f1_scores.append(np.nan)

    df["jaccard_score"] = jaccard_scores
    df["f1_score"] = f1_scores

    try:
        df.to_parquet(parquet_path)
        logging.info(
            f"Successfully updated parquet file with Jaccard and F1 scores: {parquet_path}"
        )
    except Exception as e:
        logging.error(f"Failed to save updated parquet file: {e}")


def calculate_evaluation_metrics_cropped(
    parquet_path: Path, output_path: Optional[Path] = None
):
    """
    Calculates Jaccard and F1 scores for CROPPED QA images.

    This function is designed for parquet files where crop_size is set (e.g., 64).
    It compares the segmentation mask (layer 1 of stacked image) with the
    ground truth mask, cropped to the same region using stored crop coordinates.

    Stacked image structure:
        - Layer 0: Raw microscope image crop (source image)
        - Layer 1: Segmentation mask crop (competitor's segmentation, binary 0/255)

    The GT mask is created by:
        1. Reading the full GT image
        2. Creating a binary mask for the specific cell label
        3. Cropping to the same region using crop_y_start, crop_y_end, crop_x_start, crop_x_end
        4. Padding if necessary to match crop_size

    Required parquet columns:
        - stacked_path: Path to the stacked TIFF image (can be relative to project root)
        - gt_image: Path to the ground truth image (can be relative to project root)
        - label: Cell label to extract from GT
        - crop_y_start, crop_y_end, crop_x_start, crop_x_end: Crop coordinates
        - crop_size: Size of the crop (e.g., 64)

    Args:
        parquet_path: Path to the input parquet file
        output_path: Path to save the output parquet file. If None, overwrites input file.
    """
    logging.info(f"Loading parquet file: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logging.error(f"Failed to load parquet file: {e}")
        return

    required_columns = [
        "stacked_path",
        "gt_image",
        "label",
        "crop_y_start",
        "crop_y_end",
        "crop_x_start",
        "crop_x_end",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Parquet file must contain columns: {missing_columns}")
        return

    # Find project root (directory containing 'data' folder) for resolving relative paths
    project_root = None
    parquet_abs_path = Path(parquet_path).resolve()
    for parent in parquet_abs_path.parents:
        if (parent / "data").exists():
            project_root = parent
            break

    def resolve_path(path_str):
        """Resolve a path that may be relative to project root."""
        if path_str is None:
            return None
        p = Path(path_str)
        if not p.is_absolute() and project_root:
            return project_root / p
        return p

    jaccard_scores = []
    f1_scores = []

    for index, row in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc="Calculating Evaluation Metrics (Cropped)",
    ):
        stacked_path = resolve_path(row["stacked_path"])
        gt_path = resolve_path(row["gt_image"])
        label = row["label"]

        try:
            # Read the stacked image (layer 0: raw, layer 1: segmentation)
            image_stack = tifffile.imread(stacked_path)
            if image_stack.shape[0] < 2:
                logging.warning(
                    f"Image stack has less than 2 layers, skipping: {stacked_path}"
                )
                jaccard_scores.append(np.nan)
                f1_scores.append(np.nan)
                continue

            # Get segmentation mask from layer 1 (already binary, values 0 or 255)
            seg_mask = (image_stack[1] > 0).astype(np.uint8)

            # Read the full GT image
            gt_image = tifffile.imread(gt_path)

            # Create binary mask for the specific label
            gt_full_mask = (gt_image == label).astype(np.uint8)

            # Crop the GT mask using the same coordinates as the segmentation crop
            crop_y_start = int(row["crop_y_start"])
            crop_y_end = int(row["crop_y_end"])
            crop_x_start = int(row["crop_x_start"])
            crop_x_end = int(row["crop_x_end"])
            crop_size = row.get("crop_size", None)

            gt_crop = gt_full_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            # Pad if necessary (same logic as in preprocessing)
            if crop_size is not None:
                pad_y = int(crop_size) - gt_crop.shape[0]
                pad_x = int(crop_size) - gt_crop.shape[1]
                if pad_y > 0 or pad_x > 0:
                    gt_crop = np.pad(
                        gt_crop, ((0, max(0, pad_y)), (0, max(0, pad_x))), "constant"
                    )

            gt_mask = gt_crop

            # Verify shapes match
            if gt_mask.shape != seg_mask.shape:
                logging.warning(
                    f"Shape mismatch: GT {gt_mask.shape} vs Seg {seg_mask.shape} for {stacked_path}. Skipping."
                )
                jaccard_scores.append(np.nan)
                f1_scores.append(np.nan)
                continue

            # Calculate metrics
            j_score = jaccard_score(gt_mask, seg_mask)
            f1 = f1_score(gt_mask, seg_mask)
            jaccard_scores.append(j_score)
            f1_scores.append(f1)

        except FileNotFoundError as e:
            logging.warning(
                f"File not found, skipping: {stacked_path} or {gt_path}. Error: {e}"
            )
            jaccard_scores.append(np.nan)
            f1_scores.append(np.nan)
        except Exception as e:
            logging.error(f"Error processing {stacked_path}: {e}")
            jaccard_scores.append(np.nan)
            f1_scores.append(np.nan)

    df["jaccard_score"] = jaccard_scores
    df["f1_score"] = f1_scores

    # Determine output path
    if output_path is None:
        output_path = parquet_path

    try:
        df.to_parquet(output_path)
        logging.info(
            f"Successfully saved parquet file with Jaccard and F1 scores: {output_path}"
        )
        logging.info(f"  - Mean Jaccard score: {np.nanmean(jaccard_scores):.4f}")
        logging.info(f"  - Mean F1 score: {np.nanmean(f1_scores):.4f}")
    except Exception as e:
        logging.error(f"Failed to save updated parquet file: {e}")
