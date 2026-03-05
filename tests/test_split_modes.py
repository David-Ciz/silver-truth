import pandas as pd
import sys
import os
import silver_truth.data_processing.utils.dataset_dataframe_creation as ddc

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    add_fold_split,
    add_stratified_split,
)


def test_split_modes():
    print("--- Testing Split Modes ---")

    # 1. Test Fold Logic
    # Mock DF with sequences 01 and 02, and time_frames
    data = []
    # Sequence 01: 0..9 (10 frames)
    for i in range(10):
        data.append(
            {
                "composite_key": f"01_{i:04d}.tif",
                "campaign_number": "01",
                "time_frame": i,
                "gt_image": f"dummy_01_{i}.tif",
            }
        )
    # Sequence 02: 0..9 (10 frames)
    for i in range(10):
        data.append(
            {
                "composite_key": f"02_{i:04d}.tif",
                "campaign_number": "02",
                "time_frame": i,
                "gt_image": f"dummy_02_{i}.tif",
            }
        )

    df = pd.DataFrame(data)

    # --- Test Fold-1 ---
    # Train 01, Test 02. Val ratio 0.2 (2/10 frames validation)
    print("\n[Fold-1]")
    # Train 01, Test 02. Val ratio 0.2 means split training fold 80/20.
    print("\n[Fold-1]")
    try:
        df_f1 = add_fold_split(df.copy(), "fold-1", "80,20")

        # Check Test
        test_rows = df_f1[df_f1["split"] == "test"]
        assert all(
            test_rows["campaign_number"] == "02"
        ), "Fold-1: Test set should be seq 02"
        assert len(test_rows) == 10, "Fold-1: Test set size incorrect"

        # Check Train/Val (Seq 01)
        train_rows = df_f1[df_f1["split"] == "train"]
        val_rows = df_f1[df_f1["split"] == "validation"]

        assert all(
            train_rows["campaign_number"] == "01"
        ), "Fold-1: Train set should be seq 01"
        assert all(
            val_rows["campaign_number"] == "01"
        ), "Fold-1: Val set should be seq 01"

        print(f"Train/Val Counts: {len(train_rows)} / {len(val_rows)}")
        assert len(train_rows) == 8, f"Fold-1: Expected 8 train, got {len(train_rows)}"
        assert len(val_rows) == 2, f"Fold-1: Expected 2 val, got {len(val_rows)}"

        # Temporal Check: Max train time < Min val time
        max_train_t = train_rows["time_frame"].max()
        min_val_t = val_rows["time_frame"].min()
        print(f"Temporal Split: Max Train Time={max_train_t}, Min Val Time={min_val_t}")
        assert max_train_t < min_val_t, "Fold-1: Temporal leak! Train time >= Val time"

        print("✅ Fold-1 SUCCESS")
    except Exception as e:
        print(f"❌ Fold-1 FAILED: {e}")
        raise

    # --- Test Fold-2 ---
    # Train 02, Test 01.
    # Train 02, Test 01.
    print("\n[Fold-2]")
    try:
        df_f2 = add_fold_split(df.copy(), "fold-2", "80,20")

        test_rows = df_f2[df_f2["split"] == "test"]
        assert all(
            test_rows["campaign_number"] == "01"
        ), "Fold-2: Test set should be seq 01"

        train_rows = df_f2[df_f2["split"] == "train"]
        val_rows = df_f2[df_f2["split"] == "validation"]
        assert all(
            train_rows["campaign_number"] == "02"
        ), "Fold-2: Train set should be seq 02"

        max_train_t = train_rows["time_frame"].max()
        min_val_t = val_rows["time_frame"].min()
        assert max_train_t < min_val_t, "Fold-2: Temporal leak!"

        print("✅ Fold-2 SUCCESS")
    except Exception as e:
        print(f"❌ Fold-2 FAILED: {e}")
        raise

    # --- Test Mixed (Integration with Stratification) ---
    print("\n[Mixed]")
    # Mock cell counts for stratification
    # Using direct replacement to avoid pandas+mock issues
    original_count_func = ddc.count_cells_in_image

    try:
        ddc.count_cells_in_image = lambda path: 10

        # Mixed should use 70/15/15
        df_mixed = add_stratified_split(df.copy(), "70,15,15")

        counts = df_mixed["split"].value_counts()
        print(f"Mixed Counts:\n{counts}")

        assert (
            "train" in counts and "validation" in counts and "test" in counts
        ), "Mixed: Missing splits"
        # Total 20 items. 70% = 14, 15% = 3.
        assert counts["train"] == 14, f"Mixed: Expected 14 train, got {counts['train']}"
        print("✅ Mixed SUCCESS")

    except Exception as e:
        print(f"❌ Mixed FAILED: {e}")
        # Print detailed traceback for debugging if needed, but raising should show it
        import traceback

        traceback.print_exc()
        raise
    finally:
        ddc.count_cells_in_image = original_count_func


if __name__ == "__main__":
    test_split_modes()
