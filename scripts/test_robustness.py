"""
Robustness Testing for CNN Jaccard Model

This script tests how the trained CNN model responds to degraded/noisy segmentations.
It applies various types of degradation to the segmentation masks and compares model
predictions on original vs. degraded data.

Degradation types:
- Gaussian noise
- Salt-and-pepper noise
- Pixel flipping
- Morphological operations (erosion/dilation)
- Combinations of the above
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import tifffile
from pathlib import Path
from typing import Dict, List, Tuple
import click
import logging
from dataclasses import dataclass, asdict
from scipy import ndimage
from skimage import morphology, util
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Container for robustness test results per sample"""
    
    cell_id: object  # Can be int or str
    original_prediction: float
    original_target: float
    degradation_type: str
    degradation_intensity: float
    degraded_prediction: float
    prediction_shift: float
    mae_to_target: float
    rmse_to_target: float
    confidence_change: float


class DegradedSegmentationDataset(Dataset):
    """Dataset that loads images and applies degradation to segmentation masks."""
    
    def __init__(
        self,
        parquet_file,
        data_root=None,
        transform=None,
        degradation_type='none',
        degradation_intensity=0.5
    ):
        self.data = pd.read_parquet(parquet_file)
        self.data_root = Path(data_root) if data_root else None
        self.transform = transform
        self.degradation_type = degradation_type
        self.degradation_intensity = degradation_intensity
    
    def __len__(self):
        return len(self.data)
    
    def _apply_gaussian_noise(self, mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply Gaussian noise to segmentation mask."""
        # Normalize intensity to sigma (0-1 maps to 0-0.5)
        sigma = intensity * 0.5
        noise = np.random.normal(0, sigma, mask.shape)
        degraded = np.clip(mask + noise, 0, 1)
        return degraded
    
    def _apply_salt_pepper_noise(self, mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply salt-and-pepper noise to segmentation mask."""
        degraded = util.random_noise(mask, mode='s&p', amount=intensity)
        return degraded
    
    def _apply_pixel_flipping(self, mask: np.ndarray, intensity: float) -> np.ndarray:
        """Flip random percentage of pixels in segmentation mask."""
        mask_copy = mask.copy()
        n_pixels_to_flip = int(mask.size * intensity)
        flip_indices = np.random.choice(mask.size, n_pixels_to_flip, replace=False)
        flat_mask = mask_copy.flatten()
        flat_mask[flip_indices] = 1 - flat_mask[flip_indices]
        degraded = flat_mask.reshape(mask.shape)
        return degraded
    
    def _apply_morphological_erosion(self, mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply morphological erosion to segmentation mask."""
        # Convert to binary if needed
        binary_mask = (mask > 0.5).astype(np.float32)
        # Erosion size based on intensity (1-5 pixels)
        size = max(1, int(1 + intensity * 4))
        selem = morphology.disk(size)
        degraded = morphology.binary_erosion(binary_mask, selem).astype(np.float32)
        return degraded
    
    def _apply_morphological_dilation(self, mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply morphological dilation to segmentation mask."""
        binary_mask = (mask > 0.5).astype(np.float32)
        size = max(1, int(1 + intensity * 4))
        selem = morphology.disk(size)
        degraded = morphology.binary_dilation(binary_mask, selem).astype(np.float32)
        return degraded
    
    def _apply_combined_degradation(self, mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply combination of noise and morphological operations."""
        # First apply salt-and-pepper noise
        degraded = util.random_noise(mask, mode='s&p', amount=intensity * 0.5)
        # Then apply slight erosion
        binary_mask = (degraded > 0.5).astype(np.float32)
        selem = morphology.disk(max(1, int(intensity * 2)))
        degraded = morphology.binary_erosion(binary_mask, selem).astype(np.float32)
        return degraded
    
    def _apply_blurring(self, mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply Gaussian blur to segmentation mask."""
        # Blur sigma based on intensity (0.5-3.0 pixels)
        sigma = 0.5 + intensity * 2.5
        from scipy.ndimage import gaussian_filter
        degraded = gaussian_filter(mask, sigma=sigma)
        degraded = np.clip(degraded, 0, 1)
        return degraded
    
    def _degrade_segmentation(self, mask: np.ndarray) -> np.ndarray:
        """Apply degradation to segmentation mask based on degradation_type."""
        if self.degradation_type == 'none':
            return mask
        elif self.degradation_type == 'gaussian_noise':
            return self._apply_gaussian_noise(mask, self.degradation_intensity)
        elif self.degradation_type == 'salt_pepper':
            return self._apply_salt_pepper_noise(mask, self.degradation_intensity)
        elif self.degradation_type == 'pixel_flip':
            return self._apply_pixel_flipping(mask, self.degradation_intensity)
        elif self.degradation_type == 'erosion':
            return self._apply_morphological_erosion(mask, self.degradation_intensity)
        elif self.degradation_type == 'dilation':
            return self._apply_morphological_dilation(mask, self.degradation_intensity)
        elif self.degradation_type == 'blurring':
            return self._apply_blurring(mask, self.degradation_intensity)
        elif self.degradation_type == 'combined':
            return self._apply_combined_degradation(mask, self.degradation_intensity)
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        rel_path = row["stacked_path"]
        
        if self.data_root:
            image_path = self.data_root / rel_path
        else:
            image_path = rel_path
        
        jaccard = row["jaccard_score"]
        cell_id = row.get("cell_id", idx)
        
        # Read stacked TIFF
        img_np = tifffile.imread(image_path)
        
        if img_np.ndim == 2:
            img_np = np.stack([img_np, img_np], axis=0)
        elif img_np.ndim == 3:
            if img_np.shape[0] == 2:
                pass
            elif img_np.shape[-1] == 2:
                img_np = np.transpose(img_np, (2, 0, 1))
            else:
                raise ValueError(f"Image shape not supported: {img_np.shape}")
        
        # Extract and degrade segmentation mask (channel 1)
        raw_image = img_np[0].astype(np.float32) / 255.0
        seg_mask = img_np[1].astype(np.float32) / 255.0
        
        # Apply degradation to mask
        degraded_mask = self._degrade_segmentation(seg_mask)
        
        # Stack degraded mask with original raw image
        degraded_img_np = np.stack([raw_image, degraded_mask], axis=0)
        
        # Normalize
        degraded_img_np = degraded_img_np.astype(np.float32)
        
        image = torch.from_numpy(degraded_img_np)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(jaccard, dtype=torch.float32), cell_id


def normalize_tensor(tensor, mean=[0.5, 0.5], std=[0.5, 0.5]):
    """Normalize tensor from [0, 1] to [-1, 1]."""
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


class NormalizeTransform:
    """Transform to normalize tensor to [-1, 1] from [0, 1]."""
    def __call__(self, x):
        return normalize_tensor(x, mean=[0.5, 0.5], std=[0.5, 0.5])


def get_transform():
    """Get the default transform."""
    return NormalizeTransform()


def save_degraded_images(
    degraded_mask: np.ndarray,
    cell_id: object,
    degradation_type: str,
    degradation_intensity: float,
    output_dir: str,
):
    """Save degraded mask as image."""
    output_path = Path(output_dir) / f"{degradation_type}_int{degradation_intensity:.1f}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to uint8 (0-255)
    mask_uint8 = (degraded_mask * 255).astype(np.uint8)
    
    # Save as TIFF
    filename = output_path / f"{cell_id}.tif"
    tifffile.imwrite(str(filename), mask_uint8)


def save_original_image(
    original_mask: np.ndarray,
    cell_id: object,
    output_dir: str,
):
    """Save original mask as reference."""
    output_path = Path(output_dir) / "original"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to uint8
    mask_uint8 = (original_mask * 255).astype(np.uint8)
    
    # Save as TIFF
    filename = output_path / f"{cell_id}.tif"
    if not filename.exists():  # Only save once
        tifffile.imwrite(str(filename), mask_uint8)


def get_transform():
    """Get the default transform."""
    return NormalizeTransform()


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cnn import Jaccard
    
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    
    dropout_rate = metadata.get("dropout_rate", 0.3)
    model_type = metadata.get("model_type", None)
    
    state_dict = checkpoint["model_state_dict"]
    state_dict_keys = list(state_dict.keys())
    
    # Handle key prefix mismatch (resnet -> model)
    if state_dict_keys and state_dict_keys[0].startswith("resnet."):
        logger.info("Converting checkpoint keys from 'resnet.*' to 'model.*'")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("resnet.", "model.", 1)
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        state_dict_keys = list(state_dict.keys())
    
    # If model_type not in metadata, detect from checkpoint
    if not model_type:
        # Get fc.weight shape to determine model type
        fc_weight_key = None
        for key in state_dict_keys:
            if key.endswith("fc.weight"):
                fc_weight_key = key
                break
        
        if fc_weight_key:
            fc_weight_shape = state_dict[fc_weight_key].shape[1]
            
            # Map feature size to model type
            if fc_weight_shape == 512:
                model_type = "resnet18"
            elif fc_weight_shape == 2048:
                # Could be resnet50 or resnet101 - check for layer4 structure
                if "model.layer4.2.conv2.weight" in state_dict_keys:
                    model_type = "resnet50"
                else:
                    model_type = "resnet101"  # has more layers in layer4
            elif fc_weight_shape == 1280:
                model_type = "efficientnet_b1"
            elif fc_weight_shape == 1792:
                model_type = "efficientnet_b4"
            elif fc_weight_shape == 2560:
                model_type = "efficientnet_b7"
            else:
                logger.warning(f"Unknown fc.weight size: {fc_weight_shape}, defaulting to resnet50")
                model_type = "resnet50"
            
            logger.info(f"Detected model type: {model_type} (fc features: {fc_weight_shape})")
        else:
            logger.warning("Could not find fc layer in checkpoint")
            model_type = "resnet50"
    
    model = Jaccard(dropout_rate=dropout_rate, model_type=model_type).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Model loaded successfully: {model_type}")
    return model


def run_robustness_test(
    model: torch.nn.Module,
    dataset: Dataset,
    indices: List[int],
    batch_size: int,
    device: torch.device,
    degradation_type: str,
    degradation_intensity: float,
) -> Tuple[List[RobustnessResult], np.ndarray, np.ndarray]:
    """
    Run inference on original and degraded data.
    
    Returns:
        Tuple of (results_list, predictions, actuals)
    """
    results = []
    all_predictions = []
    all_actuals = []
    all_cell_ids = []
    
    eval_subset = Subset(dataset, indices)
    eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (images, targets, cell_ids) in enumerate(eval_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            predictions = outputs.squeeze(dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_actuals.extend(targets.cpu().numpy())
            all_cell_ids.extend(cell_ids)
    
    return all_predictions, all_actuals, all_cell_ids


def extract_degraded_masks(
    parquet_file: str,
    data_root: str,
    indices: List[int],
    degradation_type: str,
    degradation_intensity: float,
    output_images_dir: str,
) -> Dict[object, np.ndarray]:
    """
    Extract degraded masks from dataset and save them.
    
    Returns:
        Dict mapping cell_id to degraded mask
    """
    data = pd.read_parquet(parquet_file)
    data_root_path = Path(data_root) if data_root else None
    
    # Create dataset with specified degradation
    dataset = DegradedSegmentationDataset(
        parquet_file,
        data_root=data_root,
        transform=None,  # No transform for mask extraction
        degradation_type=degradation_type,
        degradation_intensity=degradation_intensity,
    )
    
    masks_dict = {}
    
    # Process each index
    for idx in indices:
        try:
            row = data.iloc[idx]
            rel_path = row["stacked_path"]
            cell_id = row.get("cell_id", idx)
            
            if data_root_path:
                image_path = data_root_path / rel_path
            else:
                image_path = rel_path
            
            # Load and degrade
            img_np = tifffile.imread(image_path)
            
            if img_np.ndim == 2:
                seg_mask = img_np
            elif img_np.ndim == 3:
                if img_np.shape[0] == 2:
                    seg_mask = img_np[1]
                elif img_np.shape[-1] == 2:
                    seg_mask = img_np[:, :, 1]
                else:
                    continue
            else:
                continue
            
            # Normalize to [0, 1]
            seg_mask = seg_mask.astype(np.float32) / 255.0
            
            # Apply degradation
            degraded_mask = dataset._degrade_segmentation(seg_mask)
            masks_dict[cell_id] = degraded_mask
            
            # Save the degraded mask
            save_degraded_images(
                degraded_mask,
                cell_id,
                degradation_type,
                degradation_intensity,
                output_images_dir,
            )
        except Exception as e:
            logger.warning(f"Error processing index {idx}: {e}")
            continue
    
    return masks_dict


@click.group()
def cli():
    """Robustness testing CLI for CNN Jaccard model."""
    pass


@cli.command()
@click.option(
    "--parquet-file",
    type=click.Path(exists=True),
    default="BF-C2DL-HSC_QA_crops_64_split70-15-15_seed42.parquet",
    help="Path to the input Parquet file.",
)
@click.option(
    "--data-root",
    type=click.Path(exists=True),
    default=None,
    help="Root directory for data.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    default="efficientnet_b7_jaccard.pt",
    help="Path to the trained model checkpoint.",
)
@click.option(
    "--output-json",
    type=click.Path(),
    default="robustness_results.json",
    help="Path to save results as JSON.",
)
@click.option(
    "--output-parquet",
    type=click.Path(),
    default="robustness_results.parquet",
    help="Path to save results as Parquet.",
)
@click.option(
    "--output-images-dir",
    type=click.Path(),
    default="robustness_images",
    help="Directory to save degraded segmentation masks.",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for inference.",
)
@click.option(
    "--test-split-only",
    is_flag=True,
    default=True,
    help="Run only on test split (default: True).",
)
@click.option(
    "--num-samples",
    type=int,
    default=None,
    help="Limit number of samples to test (default: all).",
)
@click.option(
    "--save-degraded-images/--no-save-degraded-images",
    default=True,
    help="Save degraded segmentation images for visualization.",
)
@click.option(
    "--num-saved-samples",
    type=int,
    default=3,
    show_default=True,
    help="Number of samples to save as images when saving is enabled.",
)
def robustness_test(
    parquet_file,
    data_root,
    model_path,
    output_json,
    output_parquet,
    output_images_dir,
    batch_size,
    test_split_only,
    num_samples,
    save_degraded_images,
    num_saved_samples,
):
    """Test model robustness on degraded segmentations."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Load data
    data = pd.read_parquet(parquet_file)
    
    # Select split
    if test_split_only:
        indices = data[data["split"] == "test"].index.tolist()
        logger.info(f"Testing on test split: {len(indices)} samples")
    else:
        indices = list(range(len(data)))
        logger.info(f"Testing on all data: {len(indices)} samples")
    
    # Limit samples if requested
    if num_samples:
        indices = indices[:num_samples]
        logger.info(f"Limited to {num_samples} samples")
    
    # Degradation parameters to test
    degradation_configs = [
        ("none", 0.0),
        ("gaussian_noise", 0.2),
        ("gaussian_noise", 0.5),
        ("salt_pepper", 0.1),
        ("salt_pepper", 0.2),
        ("pixel_flip", 0.1),
        ("pixel_flip", 0.2),
        ("erosion", 0.3),
        ("erosion", 0.6),
        ("dilation", 0.3),
        ("dilation", 0.6),
        ("blurring", 0.3),
        ("blurring", 0.6),
        ("combined", 0.3),
        ("combined", 0.6),
    ]
    
    all_results = []
    summary_stats = {}
    
    # Original predictions (no degradation)
    logger.info("Computing predictions on original data...")
    orig_dataset = DegradedSegmentationDataset(
        parquet_file,
        data_root=data_root,
        transform=get_transform(),
        degradation_type='none',
    )
    orig_predictions, orig_targets, orig_cell_ids = run_robustness_test(
        model, orig_dataset, indices, batch_size, device, 'none', 0.0
    )
    
    orig_pred_dict = {cid: pred for cid, pred in zip(orig_cell_ids, orig_predictions)}
    orig_target_dict = {cid: tgt for cid, tgt in zip(orig_cell_ids, orig_targets)}
    
    # Save original masks as reference
    if save_degraded_images:
        logger.info("Saving original masks...")
        data = pd.read_parquet(parquet_file)
        data_root_path = Path(data_root) if data_root else None
        if num_saved_samples is not None and num_saved_samples > 0:
            image_save_indices = indices[:num_saved_samples]
        else:
            image_save_indices = indices

        for idx in image_save_indices:
            try:
                row = data.iloc[idx]
                rel_path = row["stacked_path"]
                cell_id = row.get("cell_id", idx)
                
                if data_root_path:
                    image_path = data_root_path / rel_path
                else:
                    image_path = rel_path
                
                img_np = tifffile.imread(image_path)
                if img_np.ndim == 3:
                    if img_np.shape[0] == 2:
                        seg_mask = img_np[1]
                    elif img_np.shape[-1] == 2:
                        seg_mask = img_np[:, :, 1]
                    else:
                        continue
                else:
                    continue
                
                seg_mask = seg_mask.astype(np.float32) / 255.0
                save_original_image(seg_mask, cell_id, output_images_dir)
            except Exception as e:
                logger.warning(f"Error saving original for index {idx}: {e}")
                continue
    
    # Test each degradation
    for deg_type, deg_intensity in degradation_configs:
        logger.info(f"Testing {deg_type} (intensity={deg_intensity})...")
        
        deg_dataset = DegradedSegmentationDataset(
            parquet_file,
            data_root=data_root,
            transform=get_transform(),
            degradation_type=deg_type,
            degradation_intensity=deg_intensity,
        )
        
        deg_predictions, deg_targets, deg_cell_ids = run_robustness_test(
            model, deg_dataset, indices, batch_size, device, deg_type, deg_intensity
        )
        
        # Extract and save degraded masks (for visualization)
        if save_degraded_images:
            logger.info(f"Extracting and saving degraded masks for {deg_type}...")
            extract_degraded_masks(
                parquet_file,
                data_root,
                image_save_indices,
                deg_type,
                deg_intensity,
                output_images_dir,
            )
        
        # Compare predictions
        for cid, deg_pred, deg_target in zip(deg_cell_ids, deg_predictions, deg_targets):
            orig_pred = orig_pred_dict[cid]
            orig_target = orig_target_dict[cid]
            
            prediction_shift = deg_pred - orig_pred
            mae_to_target = np.abs(deg_pred - orig_target)
            rmse_to_target = (deg_pred - orig_target) ** 2
            
            result = RobustnessResult(
                cell_id=cid if isinstance(cid, str) else int(cid),
                original_prediction=float(orig_pred),
                original_target=float(orig_target),
                degradation_type=deg_type,
                degradation_intensity=float(deg_intensity),
                degraded_prediction=float(deg_pred),
                prediction_shift=float(prediction_shift),
                mae_to_target=float(mae_to_target),
                rmse_to_target=float(rmse_to_target),
                confidence_change=float(np.abs(prediction_shift)),
            )
            
            all_results.append(result)
        
        # Compute summary statistics
        shifts = np.array([r.prediction_shift for r in all_results if r.degradation_type == deg_type])
        if len(shifts) > 0:
            summary_stats[f"{deg_type}_intensity{deg_intensity}"] = {
                "mean_shift": float(np.mean(shifts)),
                "std_shift": float(np.std(shifts)),
                "min_shift": float(np.min(shifts)),
                "max_shift": float(np.max(np.abs(shifts))),
                "n_samples": len(shifts),
            }
    
    # Save results
    logger.info(f"Saving {len(all_results)} results...")
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame([asdict(r) for r in all_results])
    
    # Save as Parquet
    results_df.to_parquet(output_parquet)
    logger.info(f"Results saved to {output_parquet}")
    
    # Save as JSON with summary
    output_data = {
        "metadata": {
            "model_path": str(model_path),
            "parquet_file": str(parquet_file),
            "device": str(device),
            "test_split_only": test_split_only,
            "num_samples": len(indices),
            "total_results": len(all_results),
        },
        "summary_statistics": summary_stats,
        "results": results_df.to_dict('records'),
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Summary saved to {output_json}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("ROBUSTNESS TEST SUMMARY")
    logger.info("="*80)
    
    for deg_type, stats in summary_stats.items():
        logger.info(f"\n{deg_type}:")
        logger.info(f"  Mean prediction shift: {stats['mean_shift']:.4f}")
        logger.info(f"  Std prediction shift:  {stats['std_shift']:.4f}")
        logger.info(f"  Min prediction shift:  {stats['min_shift']:.4f}")
        logger.info(f"  Max prediction shift:  {stats['max_shift']:.4f}")
        logger.info(f"  Samples: {stats['n_samples']}")


@cli.command()
@click.option(
    "--images-dir",
    type=click.Path(exists=True),
    default="robustness_images",
    show_default=True,
    help="Directory with degraded segmentation images (output of robustness-test).",
)
@click.option(
    "--json-file",
    type=click.Path(exists=True),
    default="robustness_summary.json",
    show_default=True,
    help="Path to the JSON summary file produced by robustness-test.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="robustness_visualizations",
    show_default=True,
    help="Directory where visualization PNGs will be saved.",
)
@click.option(
    "--max-cells",
    type=int,
    default=3,
    show_default=True,
    help="Maximum number of cells to include in the image grid (displayed horizontally).",
)
def visualize(images_dir, json_file, output_dir, max_cells):
    """Visualize degraded segmentation images and prediction-shift summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Image grid: original vs every degradation, one column per cell  #
    # ------------------------------------------------------------------ #
    subdirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if not subdirs:
        logger.warning(f"No subdirectories found in {images_dir}. Run robustness-test first.")
        return

    # Collect cell IDs from the 'original' folder (or first available folder)
    original_dir = images_dir / "original"
    ref_dir = original_dir if original_dir.exists() else subdirs[0]
    cell_files = sorted(ref_dir.glob("*.tif"))[:max_cells]
    cell_ids = [f.stem for f in cell_files]

    if not cell_ids:
        logger.warning(f"No .tif files found in {ref_dir}.")
    else:
        n_rows = len(subdirs)
        n_cols = len(cell_ids)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(max(12, n_cols * 1.8), max(6, n_rows * 1.6)),
            squeeze=False,
        )

        for row_idx, subdir in enumerate(subdirs):
            label = subdir.name
            for col_idx, cell_id in enumerate(cell_ids):
                ax = axes[row_idx][col_idx]
                tif_path = subdir / f"{cell_id}.tif"
                if tif_path.exists():
                    img = tifffile.imread(str(tif_path))
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="red")
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"cell {cell_id}", fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=60,
                                  va="center")

        fig.suptitle("Segmentation masks: original vs degradations", fontsize=11, y=1.01)
        fig.tight_layout()
        grid_path = output_dir / "image_grid.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Image grid saved to {grid_path}")

    # ------------------------------------------------------------------ #
    # 2. Bar chart: mean prediction shift per degradation type           #
    # ------------------------------------------------------------------ #
    if json_file and Path(json_file).exists():
        with open(json_file) as f:
            summary_data = json.load(f)

        stats = summary_data.get("summary_statistics", {})
        if stats:
            labels = list(stats.keys())
            mean_shifts = [stats[k]["mean_shift"] for k in labels]
            std_shifts = [stats[k]["std_shift"] for k in labels]

            colors = ["steelblue" if v >= 0 else "tomato" for v in mean_shifts]

            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
            bars = ax.bar(range(len(labels)), mean_shifts, yerr=std_shifts,
                          color=colors, capsize=4, edgecolor="black", linewidth=0.6)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel("Mean prediction shift (degraded − original)")
            ax.set_title("Robustness: mean prediction shift per degradation type")
            fig.tight_layout()

            bar_path = output_dir / "prediction_shift_chart.png"
            fig.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Prediction-shift bar chart saved to {bar_path}")
        else:
            logger.warning("No summary_statistics found in JSON file.")
    else:
        logger.warning(f"JSON file not found: {json_file}. Skipping bar chart.")

    logger.info(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    cli()
