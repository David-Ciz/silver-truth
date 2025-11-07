#!/usr/bin/env python3
"""
Script to properly visualize QA crops with all 3 layers side by side.
"""

import tifffile
import numpy as np
from pathlib import Path
import sys

def visualize_crop_simple(crop_path, output_path=None):
    """
    Visualize all 3 layers of a crop side by side and save as PNG.
    No matplotlib dependency required.
    """
    # Load the 3-layer TIFF
    img = tifffile.imread(crop_path)
    
    if len(img.shape) != 3 or img.shape[0] != 3:
        print(f"Error: Expected 3 layers, got shape: {img.shape}")
        return
    
    print(f"Loading: {Path(crop_path).name}")
    print(f"Shape: {img.shape}")
    
    # Extract layers
    source = img[0]      # Source/raw image
    fused = img[1]       # Fused segmentation
    gt = img[2]          # Ground truth
    
    print(f"\nLayer 0 (Source): min={source.min()}, max={source.max()}")
    print(f"Layer 1 (Fused):  min={fused.min()}, max={fused.max()}, non-zero={np.count_nonzero(fused)}")
    print(f"Layer 2 (GT):     min={gt.min()}, max={gt.max()}, non-zero={np.count_nonzero(gt)}")
    
    # Create side-by-side visualization
    # Normalize each layer to 0-255 for proper display
    source_norm = source  # Already in good range
    fused_norm = fused    # Already 0 or 255
    gt_norm = gt          # Already 0 or 255
    
    # Stack horizontally with small gap
    gap = np.zeros((img.shape[1], 2), dtype=np.uint8)
    combined = np.hstack([source_norm, gap, fused_norm, gap, gt_norm])
    
    # Save as PNG
    if output_path is None:
        output_path = Path(crop_path).stem + "_visualization.png"
    
    # Use PIL to save (more compatible than tifffile for PNG)
    try:
        from PIL import Image
        Image.fromarray(combined).save(output_path)
        print(f"\nSaved visualization to: {output_path}")
    except ImportError:
        # Fallback to tifffile
        tifffile.imwrite(output_path, combined)
        print(f"\nSaved visualization to: {output_path}")
    
    return combined

def batch_visualize(qa_crops_dir, output_dir, num_samples=10):
    """
    Create visualizations for multiple crops.
    """
    crops_path = Path(qa_crops_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    crop_files = sorted(list(crops_path.glob("*.tif")))[:num_samples]
    
    print(f"Creating visualizations for {len(crop_files)} crops...")
    print(f"Output directory: {output_path}\n")
    
    for crop_file in crop_files:
        output_file = output_path / f"{crop_file.stem}_viz.png"
        try:
            visualize_crop_simple(str(crop_file), str(output_file))
            print("-" * 70)
        except Exception as e:
            print(f"Error processing {crop_file.name}: {e}")

def show_layer_info(crop_path):
    """
    Just show information about the layers without creating visualization.
    """
    img = tifffile.imread(crop_path)
    
    print(f"\nFile: {Path(crop_path).name}")
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    
    if len(img.shape) == 3:
        layer_names = ["Source/Raw", "Fused Segmentation", "Ground Truth"]
        for i, name in enumerate(layer_names):
            layer = img[i]
            print(f"\nLayer {i} ({name}):")
            print(f"  Range: {layer.min()} - {layer.max()}")
            print(f"  Mean: {layer.mean():.2f}")
            print(f"  Non-zero pixels: {np.count_nonzero(layer)}")
            print(f"  Unique values: {len(np.unique(layer))}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_crops.py <crop_file.tif>           # Visualize single crop")
        print("  python visualize_crops.py qa_crops_fused --batch 5  # Visualize first 5 crops")
        print("  python visualize_crops.py <crop_file.tif> --info    # Just show layer info")
        return
    
    input_path = sys.argv[1]
    
    if "--info" in sys.argv:
        # Just show info
        show_layer_info(input_path)
    elif "--batch" in sys.argv:
        # Batch processing
        num_samples = 10
        if len(sys.argv) > sys.argv.index("--batch") + 1:
            try:
                num_samples = int(sys.argv[sys.argv.index("--batch") + 1])
            except:
                pass
        batch_visualize(input_path, "crop_visualizations", num_samples)
    else:
        # Single file
        if Path(input_path).is_file():
            visualize_crop_simple(input_path)
        else:
            print(f"Error: File not found: {input_path}")

if __name__ == "__main__":
    main()
