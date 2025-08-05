#!/usr/bin/env python3

import numpy as np
import tifffile
from PIL import Image
import sys
from pathlib import Path


def diagnose_tif(file_path):
    """Diagnose TIF file to understand why it appears black."""
    print(f"Analyzing: {file_path}")
    print("=" * 50)

    try:
        # Read with tifffile
        img = tifffile.imread(file_path)
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Min value: {np.min(img)}")
        print(f"Max value: {np.max(img)}")
        print(f"Unique values count: {len(np.unique(img))}")
        print(f"Mean value: {np.mean(img):.2f}")

        # Show some unique values
        unique_vals = np.unique(img)
        print(f"Unique label values: {unique_vals}")
        if len(unique_vals) > 10:
            print(f"First 10 values: {unique_vals[:10]}")
            print(f"Last 10 values: {unique_vals[-10:]}")

        # Check if it's a label image (typical for segmentation)
        if np.min(img) == 0 and np.max(img) > 0:
            print("\nThis appears to be a label/segmentation image!")
            print("Values represent different objects/regions:")

            # Count pixels for each label
            for label in unique_vals:
                count = np.sum(img == label)
                percentage = count / img.size * 100
                if label == 0:
                    print(
                        f"  Label {label} (background): {count:,} pixels ({percentage:.1f}%)"
                    )
                else:
                    print(
                        f"  Label {label} (object): {count:,} pixels ({percentage:.1f}%)"
                    )

            print("\nTo visualize properly, you need to:")
            print("1. Use a proper label visualization tool")
            print("2. Apply a colormap to different label values")
            print("3. Or convert to RGB with unique colors per label")

        # Check for very small values that might need scaling
        if np.max(img) < 1.0 and np.max(img) > 0:
            print(f"\nImage has very small values (max: {np.max(img)}).")
            print("It might need scaling to be visible.")

    except Exception as e:
        print(f"Error reading with tifffile: {e}")

    try:
        # Also try with PIL
        pil_img = Image.open(file_path)
        print(f"\nPIL Image mode: {pil_img.mode}")
        print(f"PIL Image size: {pil_img.size}")

    except Exception as e:
        print(f"Error reading with PIL: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_tif.py <path_to_tif_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    diagnose_tif(file_path)
