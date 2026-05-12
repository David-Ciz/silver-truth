import json

import pandas as pd

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    SPLIT_AUDIT_ATTR,
    add_fold_split,
    add_stratified_split,
)


def _build_fold_dataframe(
    gt_counts_seq01: dict[int, int], gt_counts_seq02: dict[int, int]
) -> pd.DataFrame:
    rows = []
    for campaign, counts in [("01", gt_counts_seq01), ("02", gt_counts_seq02)]:
        for time_frame in range(10):
            gt_count = counts.get(time_frame, 0)
            rows.append(
                {
                    "composite_key": f"{campaign}_{time_frame:04d}.tif",
                    "campaign_number": campaign,
                    "time_frame": time_frame,
                    "gt_image": f"{campaign}_gt_{time_frame:04d}.tif"
                    if gt_count
                    else None,
                    "has_gt": gt_count > 0,
                    "gt_cell_count": gt_count,
                }
            )
    return pd.DataFrame(rows)


def test_fold_split_balances_supervised_cells_and_preserves_sequences() -> None:
    df = _build_fold_dataframe(
        gt_counts_seq01={1: 10, 3: 10, 5: 25, 7: 25, 9: 30},
        gt_counts_seq02={0: 12, 2: 12, 4: 12},
    )

    split_df = add_fold_split(df.copy(), "fold-1", "80,20")
    audit = json.loads(split_df.attrs[SPLIT_AUDIT_ATTR])

    assert set(split_df.loc[split_df["split"] == "test", "campaign_number"]) == {"02"}
    assert set(split_df.loc[split_df["split"] != "test", "campaign_number"]) == {"01"}
    assert audit["validation_selection_mode"] == "contiguous_gt_block"
    assert audit["whole_image_hard_constraints_passed"] is True
    assert audit["validation_gt_cells"] == 20
    assert audit["train_gt_cells"] == 80
    assert audit["actual_val_fraction_by_gt_cells"] == 0.2
    assert set(audit["selected_validation_gt_keys"]) == {"01_0001.tif", "01_0003.tif"}
    assert {"has_gt", "gt_cell_count", "split"}.issubset(split_df.columns)


def test_fold_split_can_fall_back_to_non_contiguous_gt_subset() -> None:
    df = _build_fold_dataframe(
        gt_counts_seq01={0: 30, 1: 5, 2: 30, 3: 5},
        gt_counts_seq02={0: 10, 1: 10, 2: 10},
    )

    split_df = add_fold_split(df.copy(), "fold-1", "80,20")
    audit = json.loads(split_df.attrs[SPLIT_AUDIT_ATTR])

    assert audit["validation_selection_mode"] == "subset_sum_fallback"
    assert set(audit["selected_validation_gt_keys"]) == {"01_0001.tif", "01_0003.tif"}
    validation_gt = split_df[(split_df["split"] == "validation") & (split_df["has_gt"])]
    assert set(validation_gt["composite_key"]) == {"01_0001.tif", "01_0003.tif"}
    assert audit["whole_image_hard_constraints_passed"] is True


def test_mixed_split_preserves_supervised_columns() -> None:
    df = _build_fold_dataframe(
        gt_counts_seq01={0: 10, 1: 10, 2: 10, 3: 10, 4: 10},
        gt_counts_seq02={0: 10, 1: 10, 2: 10, 3: 10, 4: 10},
    )

    split_df = add_stratified_split(df.copy(), "70,15,15")

    counts = split_df["split"].value_counts()
    assert {"train", "validation", "test"}.issubset(set(counts.index))
    assert split_df["has_gt"].dtype == bool
    assert split_df["gt_cell_count"].sum() == 100
