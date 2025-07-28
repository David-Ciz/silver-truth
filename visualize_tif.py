#!/usr/bin/env python3

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
import random

def visualize_segmentation_mask(file_path, save_png=True):
    """Visualize segmentation mask with proper coloring."""
    print(f"Visualizing: {file_path}")
    
    # Read the mask
    mask = tifffile.imread(file_path)
    
    print(f"Image shape: {mask.shape}")
    print(f"Unique labels: {np.unique(mask)}")
    print(f"Number of objects: {len(np.unique(mask)) - 1}")  # -1 to exclude background
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original mask (binary)
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Original Mask (Binary)')
    axes[0].axis('off')
    
    # 2. Mask with custom colormap
    # Create a colormap that makes objects visible
    if mask.max() > 0:
        # For binary masks, use a colormap that shows objects clearly
        custom_mask = mask.astype(float)
        custom_mask[custom_mask == 0] = 0  # Background stays 0
        custom_mask[custom_mask > 0] = 1   # Objects become 1
        
        axes[1].imshow(custom_mask, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Objects Highlighted')
        axes[1].axis('off')
        
        # 3. Contours overlay
        axes[2].imshow(mask == 0, cmap='gray', alpha=0.7)  # Background
        axes[2].contour(mask, levels=[0.5], colors='red', linewidths=1)
        axes[2].set_title('Object Contours')
        axes[2].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No objects found', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')
        axes[2].text(0.5, 0.5, 'No objects found', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_png:
        output_path = Path(file_path).with_suffix('.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    return mask

def visualize_multiple_labels(file_path, save_png=True):
    """Visualize segmentation with multiple labels (each object different color)."""
    mask = tifffile.imread(file_path)
    unique_labels = np.unique(mask)
    
    if len(unique_labels) <= 2:  # Only background and one label
        return visualize_segmentation_mask(file_path, save_png)
    
    print(f"Found {len(unique_labels)} different labels: {unique_labels}")
    
    # Create a properly mapped colormap
    # We need to remap the labels to consecutive indices for proper coloring
    mask_remapped = np.zeros_like(mask)
    
    # Create color mapping - each unique label gets a unique color
    n_colors = len(unique_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, n_colors))
    
    # Ensure background (label 0) is black
    bg_idx = np.where(unique_labels == 0)[0][0]
    colors[bg_idx] = [0, 0, 0, 1]  # Black for background
    
    # Remap labels to consecutive indices
    for new_idx, original_label in enumerate(unique_labels):
        mask_remapped[mask == original_label] = new_idx
    
    cmap = mcolors.ListedColormap(colors)
    
    # Create figure with multiple views
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original mapping with proper colormap
    im1 = axes[0].imshow(mask_remapped, cmap=cmap, vmin=0, vmax=n_colors-1)
    axes[0].set_title(f'Segmentation with {len(unique_labels)-1} objects\n(Remapped for visualization)')
    axes[0].axis('off')
    
    # Add colorbar with original label values
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    # Create custom tick labels showing original label values
    tick_positions = np.arange(n_colors)
    cbar1.set_ticks(tick_positions)
    cbar1.set_ticklabels([f'Label {label}' for label in unique_labels])
    cbar1.set_label('Original Label Values')
    
    # Contour overlay
    axes[1].imshow(mask == 0, cmap='gray', alpha=0.7)  # Background
    contour_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)-1))  # Exclude background
    
    # Draw contours for each object with different colors
    legend_handles = []
    for i, label in enumerate(unique_labels[1:]):  # Skip background (0)
        if np.any(mask == label):
            contour = axes[1].contour(mask == label, levels=[0.5], 
                          colors=[contour_colors[i]], linewidths=2)
            # Create legend handle manually
            legend_handles.append(plt.Line2D([0], [0], color=contour_colors[i], 
                                           linewidth=2, label=f'Object {label}'))
    
    axes[1].set_title('Object Boundaries')
    axes[1].axis('off')
    if legend_handles:
        axes[1].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_png:
        output_path = Path(file_path).with_suffix('.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    return mask

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_tif.py <path_to_tif_file>")
        print("This will create a PNG visualization of the segmentation mask.")
        sys.exit(1)
        
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    try:
        mask = visualize_multiple_labels(file_path)
        
        # Additional statistics
        print(f"\nStatistics:")
        print(f"Total pixels: {mask.size}")
        print(f"Background pixels: {np.sum(mask == 0)}")
        print(f"Object pixels: {np.sum(mask > 0)}")
        print(f"Object coverage: {np.sum(mask > 0) / mask.size * 100:.1f}%")
        
    except Exception as e:
        print(f"Error visualizing file: {e}")
        import traceback
        traceback.print_exc()
