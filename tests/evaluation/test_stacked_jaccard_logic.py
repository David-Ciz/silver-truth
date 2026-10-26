from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silver_truth.evaluation.stacked_jaccard_logic import (
    calculate_evaluation_metrics_cropped,
)
from silver_truth.metrics.metrics import crop_with_padding

tifffile = pytest.importorskip("tifffile")


def test_calculate_evaluation_metrics_cropped_handles_negative_crop_offsets(
    tmp_path: Path,
) -> None:
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[0:2, 0:2] = 1
    gt_path = tmp_path / "gt.tif"
    tifffile.imwrite(gt_path, gt)

    seg_crop = np.zeros((4, 4), dtype=np.uint8)
    seg_crop[1:3, 1:3] = 255
    stacked = np.stack([np.zeros_like(seg_crop), seg_crop], axis=0)
    stacked_path = tmp_path / "stacked.tif"
    tifffile.imwrite(stacked_path, stacked)

    parquet_path = tmp_path / "qa.parquet"
    pd.DataFrame(
        [
            {
                "stacked_path": str(stacked_path),
                "gt_image": str(gt_path),
                "label": 1,
                "crop_y_start": -1,
                "crop_y_end": 3,
                "crop_x_start": -1,
                "crop_x_end": 3,
                "crop_size": 4,
            }
        ]
    ).to_parquet(parquet_path)

    calculate_evaluation_metrics_cropped(parquet_path)
    out_df = pd.read_parquet(parquet_path)

    assert out_df.loc[0, "jaccard_score"] == pytest.approx(1.0)
    assert out_df.loc[0, "f1_score"] == pytest.approx(1.0)


def test_crop_with_padding_handles_lower_right_boundary() -> None:
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)

    crop = crop_with_padding(image, 2, 6, 2, 6)

    expected = np.array(
        [
            [10, 11, 0, 0],
            [14, 15, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(crop, expected)
