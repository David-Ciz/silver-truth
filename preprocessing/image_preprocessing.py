import numpy as np
import tifffile
from sklearn.metrics import jaccard_score
from typing import Optional


def merge_source_with_mask(mask_file: str, gt_source_file: str, gt_mask_file: str, label: int,
                           j_value: float) -> np.ndarray:
    """
    Merge source image with ground truth and predicted masks.

    Args:
        mask_file (str): Path to the predicted mask file.
        gt_source_file (str): Path to the ground truth source image file.
        gt_mask_file (str): Path to the ground truth mask file.
        label (int): Label to use for masking.
        j_value (float): Expected Jaccard score.

    Returns:
        np.ndarray: Merged image with source and predicted mask.

    Raises:
        Exception: If no label mapping is found.
    """
    gt_source = tifffile.imread(gt_source_file)
    gt_masks = tifffile.imread(gt_mask_file)
    gt_mask = gt_masks == label
    pred_masks = tifffile.imread(mask_file)
    pred_mask = get_label_mask(pred_masks, gt_mask, label, j_value)

    if pred_mask is None:
        raise Exception(f"No label mapping found for {mask_file}, with label {label}")
    else:
        return np.stack([gt_source, pred_mask], axis=0)


def get_label_mask(pred_masks: np.ndarray, gt_mask: np.ndarray, label: int, j_value: float) -> Optional[np.ndarray]:
    """
    Get the predicted mask for a specific label.

    Args:
        pred_masks (np.ndarray): Predicted masks.
        gt_mask (np.ndarray): Ground truth mask.
        label (int): Label to use for masking.
        j_value (float): Expected Jaccard score.

    Returns:
        Optional[np.ndarray]: Predicted mask for the given label, or None if not found.
    """
    unique_mask_labels = np.unique(pred_masks)

    # Try if the masks are sorted correctly
    if label in unique_mask_labels:
        exp_pred_mask = pred_masks == label
        calculated_j_value = np.round(jaccard_score(gt_mask, exp_pred_mask, average="micro"), 6)
        if calculated_j_value == j_value:
            return exp_pred_mask

    # If the masks are not sorted correctly, try every mask
    for mask_label in unique_mask_labels[1:]:
        exp_pred_mask = pred_masks == mask_label
        calculated_j_value = np.round(jaccard_score(gt_mask, exp_pred_mask, average="micro"), 6)
        if calculated_j_value == j_value:
            return exp_pred_mask

    return None
