import numpy as np
from sklearn.metrics import jaccard_score
from scipy.ndimage import find_objects
import logging


def crop_with_padding(
    image: np.ndarray,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Crop a 2D image using possibly out-of-bounds coordinates.

    Regions outside the source image are zero-padded in the returned crop. This
    mirrors QA crop preprocessing, where boundary crops retain their requested
    size and use zero padding outside image bounds.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape={image.shape}")

    if output_shape is None:
        out_h = max(0, y_end - y_start)
        out_w = max(0, x_end - x_start)
    else:
        out_h, out_w = output_shape

    crop = np.zeros((out_h, out_w), dtype=image.dtype)
    if out_h == 0 or out_w == 0:
        return crop

    src_y0 = max(0, y_start)
    src_x0 = max(0, x_start)
    src_y1 = min(image.shape[0], y_end)
    src_x1 = min(image.shape[1], x_end)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return crop

    dst_y0 = max(0, -y_start)
    dst_x0 = max(0, -x_start)
    copy_h = min(src_y1 - src_y0, out_h - dst_y0)
    copy_w = min(src_x1 - src_x0, out_w - dst_x0)

    if copy_h <= 0 or copy_w <= 0:
        return crop

    crop[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = image[
        src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
    ]
    return crop


def calculate_labelwise_scores(gt_image, mask_image):
    """
    Calculate per-label IoU and F1 scores for a labeled segmentation mask.

    Returns
    -------
    dict
        Mapping of label -> {"jaccard": float, "f1": float}.
    """
    labels = np.unique(gt_image)[1:]  # Exclude background (0)
    scores = {}
    for label in labels:
        label_layer = np.zeros_like(gt_image)
        label_layer[gt_image == label] = 1
        mask_layer = np.zeros_like(mask_image)
        mask_layer[mask_image == label] = 1

        jaccard = jaccard_score(label_layer, mask_layer, average="micro")

        intersection = np.logical_and(label_layer, mask_layer).sum()
        label_sum = label_layer.sum()
        mask_sum = mask_layer.sum()
        denominator = label_sum + mask_sum
        f1 = float((2 * intersection) / denominator) if denominator > 0 else 1.0

        scores[label] = {"jaccard": float(jaccard), "f1": f1}
    return scores


def calculate_jaccard_scores(gt_image, mask_image):
    return {
        label: metric_values["jaccard"]
        for label, metric_values in calculate_labelwise_scores(
            gt_image=gt_image, mask_image=mask_image
        ).items()
    }


def calculate_qa_jaccard_score(
    gt_image, predicted_mask, target_label, original_image_key, campaign, qa_row
):
    """
    Calculate Jaccard score for QA cropped images.

    Uses crop coordinates stored in QA metadata to extract the exact GT region
    that corresponds to the cropped predicted mask.

    Args:
        gt_image: Full ground truth segmentation image
        predicted_mask: Binary mask from the cropped stacked image (0s and 1s)
        target_label: The cell label we're evaluating
        original_image_key: Key to identify the original image (e.g., "t0061")
        campaign: Campaign number
        qa_row: Row from QA dataframe with metadata including crop coordinates

    Returns:
        Jaccard score (float) or None if calculation fails
    """
    try:
        # Check if we have crop coordinate information
        has_crop_coords = all(
            col in qa_row
            for col in ["crop_y_start", "crop_y_end", "crop_x_start", "crop_x_end"]
        )

        if has_crop_coords and qa_row.get("crop_size") is not None:
            # Use stored crop coordinates
            y_start = int(qa_row["crop_y_start"])
            y_end = int(qa_row["crop_y_end"])
            x_start = int(qa_row["crop_x_start"])
            x_end = int(qa_row["crop_x_end"])

            gt_full_mask = (gt_image == target_label).astype(np.uint8)
            gt_mask = crop_with_padding(
                gt_full_mask,
                y_start,
                y_end,
                x_start,
                x_end,
                output_shape=predicted_mask.shape,
            )

        elif predicted_mask.shape == gt_image.shape:
            # Full image case - extract GT mask for the target label
            gt_mask = (gt_image == target_label).astype(np.uint8)

        else:
            # Fallback: try to find the best matching region
            logging.warning(
                f"No crop coordinates available for {original_image_key}, using fallback method"
            )

            gt_mask_full = (gt_image == target_label).astype(np.uint8)

            if np.sum(gt_mask_full) == 0:
                # No target label in GT
                return 0.0

            # Find all connected components of the target label
            labeled_gt = (gt_image == target_label).astype(int)
            objects = find_objects(labeled_gt)

            if not objects:
                return 0.0

            # Use the first object (in a more sophisticated implementation,
            # we'd find the best matching crop based on size/position)
            slice_y, slice_x = objects[0]

            # Extract the region
            gt_region = gt_mask_full[slice_y, slice_x]

            # Resize to match predicted_mask size
            target_h, target_w = predicted_mask.shape
            gt_h, gt_w = gt_region.shape

            if gt_h == target_h and gt_w == target_w:
                gt_mask = gt_region
            else:
                # Simple center alignment
                gt_mask = np.zeros((target_h, target_w), dtype=np.uint8)

                # Calculate offsets for centering
                start_y = max(0, (gt_h - target_h) // 2)
                start_x = max(0, (gt_w - target_w) // 2)
                end_y = min(gt_h, start_y + target_h)
                end_x = min(gt_w, start_x + target_w)

                dest_start_y = max(0, (target_h - gt_h) // 2)
                dest_start_x = max(0, (target_w - gt_w) // 2)
                dest_end_y = dest_start_y + (end_y - start_y)
                dest_end_x = dest_start_x + (end_x - start_x)

                gt_mask[dest_start_y:dest_end_y, dest_start_x:dest_end_x] = gt_region[
                    start_y:end_y, start_x:end_x
                ]

        # Calculate Jaccard score
        intersection = np.sum(gt_mask & predicted_mask)
        union = np.sum(gt_mask | predicted_mask)

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        jaccard = intersection / union
        return jaccard

    except Exception as e:
        logging.error(
            f"Error calculating QA Jaccard score for {original_image_key}, label {target_label}: {e}"
        )
        return None
