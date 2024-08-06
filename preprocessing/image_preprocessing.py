import numpy as np
import tifffile
from sklearn.metrics import jaccard_score
from typing import Optional, Set, Tuple


def merge_source_with_mask(
    pred_masks: np.ndarray,
    gt_source: np.ndarray,
    gt_mask: np.ndarray,
    label: int,
    j_value: float,
    used_labels: Set[int]
) -> Tuple[np.ndarray, int]:
    """
    Merge source image with ground truth and predicted masks.

    Args:
        pred_masks (np.ndarray): Predicted masks array.
        gt_source (np.ndarray): Ground truth source image array.
        gt_mask (np.ndarray): Ground truth mask array.
        label (int): Label to use for masking.
        j_value (float): Expected Jaccard score.
        used_labels (Set[int]): Set of already used labels.

    Returns:
        Tuple[np.ndarray, int]: Merged image with source and predicted mask, and the used label.

    Raises:
        Exception: If no label mapping is found.
    """
    pred_mask, used_label = get_label_mask(pred_masks, gt_mask, label, j_value, used_labels)
    if pred_mask is None:
        raise Exception(f"No label mapping found for {label}")
    else:
        return np.stack([gt_source, pred_mask], axis=0), used_label


def get_label_mask(
    pred_masks: np.ndarray,
    gt_mask: np.ndarray,
    label: int,
    j_value: float,
    used_labels: Set[int]
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Get the predicted mask for a specific label.

    Args:
        pred_masks (np.ndarray): Predicted masks array.
        gt_mask (np.ndarray): Ground truth mask array.
        label (int): Label to use for masking.
        j_value (float): Expected Jaccard score.
        used_labels (Set[int]): Set of already used labels.

    Returns:
        Tuple[Optional[np.ndarray], Optional[int]]: Predicted mask for the given label and the used label,
        or (None, None) if not found.
    """
    unique_mask_labels = np.unique(pred_masks)
    unique_mask_labels = np.setdiff1d(unique_mask_labels, list(used_labels))

    # Try the given label first
    if label in unique_mask_labels:
        exp_pred_mask = pred_masks == label
        calculated_j_value = jaccard_score(gt_mask, exp_pred_mask, average="micro")
        if np.isclose(calculated_j_value, j_value, rtol=1e-6):
            return exp_pred_mask, label

    # If not found, try other labels
    for mask_label in unique_mask_labels[unique_mask_labels != 0]:
        exp_pred_mask = pred_masks == mask_label
        calculated_j_value = jaccard_score(gt_mask, exp_pred_mask, average="micro")
        if np.isclose(calculated_j_value, j_value, rtol=1e-6):
            return exp_pred_mask, mask_label

    return None, None
