import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    add_stratified_split,
)


def test_stratification():
    # create dummy dataframe
    # We want 100 images with varying cell counts to test distribution
    data = []

    # Let's create a distribution of cell counts.
    # 5 images with 100 cells
    # 20 images with 10 cells
    # 75 images with 1 cell
    # Total cells = 5*100 + 20*10 + 75*1 = 500 + 200 + 75 = 775 cells

    # Expected (70/15/15):
    # Train: ~542.5 cells
    # Val: ~116.25 cells
    # Test: ~116.25 cells

    images_counts = {}

    for i in range(5):
        images_counts[f"img_large_{i}.tif"] = 100
        data.append({"gt_image": f"img_large_{i}.tif", "id": f"large_{i}"})

    for i in range(20):
        images_counts[f"img_medium_{i}.tif"] = 10
        data.append({"gt_image": f"img_medium_{i}.tif", "id": f"medium_{i}"})

    for i in range(75):
        images_counts[f"img_small_{i}.tif"] = 1
        data.append({"gt_image": f"img_small_{i}.tif", "id": f"small_{i}"})

    df = pd.DataFrame(data)

    # Mock count_cells_in_image
    # We define a side_effect function for the mock
    def mock_imread(path):
        count = images_counts.get(path, 0)
        # return array with 'count' unique non-zero values
        # we can just return a range(count+1) so unique is count+1 (including 0)
        return np.arange(count + 1)

    with patch(
        "silver_truth.data_processing.utils.dataset_dataframe_creation.tifffile.imread",
        side_effect=mock_imread,
    ):
        split_ratios = "70,15,15"
        result_df = add_stratified_split(df.copy(), split_ratios)

        # Verify
        print("--- Verification Results ---")
        if "split" not in result_df.columns:
            print("FAILED: 'split' column missing.")
            return

        train_df = result_df[result_df["split"] == "train"]
        val_df = result_df[result_df["split"] == "validation"]
        test_df = result_df[result_df["split"] == "test"]

        train_cells = sum([images_counts[x] for x in train_df["gt_image"]])
        val_cells = sum([images_counts[x] for x in val_df["gt_image"]])
        test_cells = sum([images_counts[x] for x in test_df["gt_image"]])

        total = 775
        print(f"Total Cells: {total}")
        print(f"Train: {train_cells} ({train_cells/total:.2%}) - Target 70%")
        print(f"Val:   {val_cells} ({val_cells/total:.2%}) - Target 15%")
        print(f"Test:  {test_cells} ({test_cells/total:.2%}) - Target 15%")

        # Assertions (approximate)
        # Using greedy can be slightly off but should be close
        assert 0.65 < train_cells / total < 0.75, "Train split off"
        assert 0.10 < val_cells / total < 0.20, "Val split off"
        assert 0.10 < test_cells / total < 0.20, "Test split off"

        print("SUCCESS: Splits are within acceptable range.")


if __name__ == "__main__":
    test_stratification()
