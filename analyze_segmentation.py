#!/usr/bin/env python3

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
import colorsys

def generate_distinct_colors(n, exclude_black=True):
    """Generate n visually distinct colors."""
    colors = []
    
    if exclude_black:
        colors.append([0, 0, 0, 1])  # Black for background
        n = n - 1
    
    # Generate colors using HSV color space for better distribution
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.8 + (i % 2) * 0.2       # Vary brightness slightly
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([rgb[0], rgb[1], rgb[2], 1.0])
    
    return colors

def visualize_with_distinct_colors(file_path, save_png=True):
    """Visualize segmentation with maximally distinct colors for each object."""
    mask = tifffile.imread(file_path)
    unique_labels = np.unique(mask)
    
    print(f"Processing: {file_path}")
    print(f"Found labels: {unique_labels}")
    print(f"Number of objects: {len(unique_labels) - 1}")  # -1 for background
    
    # Generate distinct colors
    colors = generate_distinct_colors(len(unique_labels))
    
    # Create remapped mask for proper visualization
    mask_remapped = np.zeros_like(mask, dtype=np.uint8)
    
    # Map each unique label to a consecutive index
    for new_idx, original_label in enumerate(unique_labels):
        mask_remapped[mask == original_label] = new_idx
    
    # Create custom colormap
    cmap = mcolors.ListedColormap(colors)
    
    # Create simple single-panel visualization
    plt.figure(figsize=(12, 10))
    
    # Show colored segmentation
    plt.imshow(mask_remapped, cmap=cmap, vmin=0, vmax=len(unique_labels)-1)
    plt.title(f'Segmentation: {Path(file_path).name}\n{len(unique_labels)-1} objects detected', 
              fontsize=14, pad=20)
    plt.axis('off')
    
    # Add colorbar with original label values
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    tick_positions = np.arange(len(unique_labels))
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f'BG' if label == 0 else f'{label}' for label in unique_labels])
    cbar.set_label('Object Labels', fontsize=12)
    
    plt.tight_layout()
    
    if save_png:
        # Create output filename with _colored suffix
        stem = Path(file_path).stem
        parent = Path(file_path).parent
        output_path = parent / f"{stem}_colored.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Colored segmentation saved to: {output_path}")
    
    plt.show()
    
    return mask

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_segmentation.py <path_to_tif_file>")
        print("Creates a comprehensive analysis of segmentation masks with distinct colors.")
        sys.exit(1)
        
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    try:
        visualize_with_distinct_colors(file_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
