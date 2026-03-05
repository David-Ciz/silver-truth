from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from silver_truth.ensemble import reconstruction


tifffile = pytest.importorskip("tifffile")


def test_reconstruct_full_images_from_arrays_places_cells_and_scores(
    tmp_path: Path,
) -> None:
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[1:3, 1:3] = 1
    gt[5:7, 5:7] = 2
    gt_path = tmp_path / "gt.tif"
    tifffile.imwrite(gt_path, gt)

    df = pd.DataFrame(
        [
            {
                "campaign_number": "01",
                "original_image_key": "t0001",
                "gt_image": str(gt_path),
                "split": "test",
                "recon_crop_y_start": 1,
                "recon_crop_y_end": 3,
                "recon_crop_x_start": 1,
                "recon_crop_x_end": 3,
            },
            {
                "campaign_number": "01",
                "original_image_key": "t0001",
                "gt_image": str(gt_path),
                "split": "test",
                "recon_crop_y_start": 5,
                "recon_crop_y_end": 7,
                "recon_crop_x_start": 5,
                "recon_crop_x_end": 7,
            },
        ]
    )

    pred_a = np.ones((2, 2), dtype=np.float32)
    pred_b = np.ones((2, 2), dtype=np.float32)

    out_df = reconstruction.reconstruct_full_images_from_arrays(
        databank_df=df,
        predicted_crops=[pred_a, pred_b],
        output_dir=tmp_path / "reconstructed",
        threshold=0.5,
    )

    assert len(out_df) == 1
    assert float(out_df.loc[0, "iou"]) == 1.0
    assert float(out_df.loc[0, "f1"]) == 1.0
    assert Path(out_df.loc[0, "reconstructed_path"]).exists()


def test_reconstruct_full_images_from_arrays_handles_negative_offsets(
    tmp_path: Path,
) -> None:
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[0:2, 0:2] = 1
    gt_path = tmp_path / "gt_edge.tif"
    tifffile.imwrite(gt_path, gt)

    df = pd.DataFrame(
        [
            {
                "campaign_number": "01",
                "original_image_key": "t0002",
                "gt_image": str(gt_path),
                "split": "validation",
                "recon_crop_y_start": -1,
                "recon_crop_y_end": 1,
                "recon_crop_x_start": -1,
                "recon_crop_x_end": 1,
            }
        ]
    )
    pred = np.ones((2, 2), dtype=np.float32)

    out_df = reconstruction.reconstruct_full_images_from_arrays(
        databank_df=df,
        predicted_crops=[pred],
        output_dir=tmp_path / "reconstructed_edge",
        threshold=0.5,
    )

    assert len(out_df) == 1
    assert 0.0 <= float(out_df.loc[0, "iou"]) <= 1.0
