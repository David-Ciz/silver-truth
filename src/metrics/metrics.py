import numpy as np
from sklearn.metrics import jaccard_score
from scipy.ndimage import find_objects
import tifffile
from pathlib import Path
import logging


def calculate_jaccard_scores(gt_image, mask_image):
    labels = np.unique(gt_image)[1:]  # Exclude background (0)
    scores = {}
    for label in labels:
        label_layer = np.zeros_like(gt_image)
        label_layer[gt_image == label] = 1
        mask_layer = np.zeros_like(mask_image)
        mask_layer[mask_image == label] = 1
        j = jaccard_score(label_layer, mask_layer, average="micro")
        scores[label] = j
    return scores


def calculate_dice_coefficient(gt_mask, pred_mask):
    """
    Calculate Dice coefficient (F1 score) between two binary masks.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        gt_mask: Ground truth binary mask (numpy array)
        pred_mask: Predicted binary mask (numpy array)
        
    Returns:
        Dice coefficient (float) between 0 and 1
    """
    intersection = np.sum(gt_mask & pred_mask)
    sum_masks = np.sum(gt_mask) + np.sum(pred_mask)
    
    if sum_masks == 0:
        # Both masks are empty
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / sum_masks
    return dice


def calculate_dice_scores(gt_image, mask_image):
    """
    Calculate Dice coefficient for each label in the images.
    
    Args:
        gt_image: Ground truth segmentation image
        mask_image: Predicted segmentation image
        
    Returns:
        Dictionary mapping label -> Dice score
    """
    labels = np.unique(gt_image)[1:]  # Exclude background (0)
    scores = {}
    for label in labels:
        label_layer = (gt_image == label).astype(np.uint8)
        mask_layer = (mask_image == label).astype(np.uint8)
        dice = calculate_dice_coefficient(label_layer, mask_layer)
        scores[label] = dice
    return scores


def calculate_qa_jaccard_score(gt_image, predicted_mask, target_label, original_image_key, campaign, qa_row):
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
        has_crop_coords = all(col in qa_row for col in ['crop_y_start', 'crop_y_end', 'crop_x_start', 'crop_x_end'])
        
        if has_crop_coords and qa_row.get('crop_size') is not None:
            # Use stored crop coordinates
            y_start = int(qa_row['crop_y_start'])
            y_end = int(qa_row['crop_y_end'])
            x_start = int(qa_row['crop_x_start'])
            x_end = int(qa_row['crop_x_end'])
            
            # Extract GT region that corresponds to the crop
            gt_region = gt_image[y_start:y_end, x_start:x_end]
            gt_mask = (gt_region == target_label).astype(np.uint8)
            
            # Ensure dimensions match (handle padding that might have been applied)
            if gt_mask.shape != predicted_mask.shape:
                # Resize to match predicted_mask
                target_h, target_w = predicted_mask.shape
                gt_h, gt_w = gt_mask.shape
                
                if gt_h <= target_h and gt_w <= target_w:
                    # GT region is smaller, pad it
                    pad_y = target_h - gt_h
                    pad_x = target_w - gt_w
                    gt_mask = np.pad(gt_mask, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)
                else:
                    # GT region is larger, crop it
                    gt_mask = gt_mask[:target_h, :target_w]
            
        elif predicted_mask.shape == gt_image.shape:
            # Full image case - extract GT mask for the target label
            gt_mask = (gt_image == target_label).astype(np.uint8)
            
        else:
            # Fallback: try to find the best matching region
            logging.warning(f"No crop coordinates available for {original_image_key}, using fallback method")
            
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
                
                gt_mask[dest_start_y:dest_end_y, dest_start_x:dest_end_x] = \
                    gt_region[start_y:end_y, start_x:end_x]
        
        # Calculate Jaccard score
        intersection = np.sum(gt_mask & predicted_mask)
        union = np.sum(gt_mask | predicted_mask)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        jaccard = intersection / union
        return jaccard
        
    except Exception as e:
        logging.error(f"Error calculating QA Jaccard score for {original_image_key}, label {target_label}: {e}")
        return None
