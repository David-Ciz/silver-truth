from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silver_truth.ensemble import databanks_builds as db_builds
from silver_truth.ensemble.datasets import Version

tifffile = pytest.importorskip("tifffile")


def _write_crop_stack(path: Path, mask: np.ndarray) -> None:
    raw = np.zeros(mask.shape, dtype=np.uint8)
    gt = np.zeros(mask.shape, dtype=np.uint8)
    tra = np.zeros(mask.shape, dtype=np.uint8)
    stacked = np.stack([raw, mask.astype(np.uint8), gt, tra], axis=0)
    tifffile.imwrite(path, stacked)


def test_build_databank_c3_writes_consensus_summary_channels(
    tmp_path: Path, monkeypatch
) -> None:
    gt = np.zeros((12, 12), dtype=np.uint8)
    gt[4:8, 4:8] = 1
    gt_path = tmp_path / "gt_cell_c3.tif"
    tifffile.imwrite(gt_path, gt)

    mask_a = np.zeros((4, 4), dtype=np.uint8)
    mask_a[1:3, 1:3] = 255
    mask_b = np.zeros((4, 4), dtype=np.uint8)
    mask_b[1:3, 2:4] = 255

    stack_a = tmp_path / "cell_c3_a.tif"
    stack_b = tmp_path / "cell_c3_b.tif"
    _write_crop_stack(stack_a, mask_a)
    _write_crop_stack(stack_b, mask_b)

    qa_df = pd.DataFrame(
        [
            {
                "cell_id": "c01_t0001_compA_1",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "compA",
                "stacked_path": str(stack_a),
                "gt_image": str(gt_path),
                "split": "train",
                "crop_size": 4,
                "crop_y_start": 4,
                "crop_y_end": 8,
                "crop_x_start": 4,
                "crop_x_end": 8,
                "original_center_y": 6,
                "original_center_x": 6,
            },
            {
                "cell_id": "c01_t0001_compB_1",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "compB",
                "stacked_path": str(stack_b),
                "gt_image": str(gt_path),
                "split": "train",
                "crop_size": 4,
                "crop_y_start": 4,
                "crop_y_end": 8,
                "crop_x_start": 4,
                "crop_x_end": 8,
                "original_center_y": 6,
                "original_center_x": 6,
            },
        ]
    )
    qa_parquet = tmp_path / "qa_cell_c3.parquet"
    qa_df.to_parquet(qa_parquet)

    monkeypatch.setattr(db_builds.ext, "compress_images", lambda *args, **kwargs: None)

    build_opt = {
        "name": "BF-C2DL-HSC",
        "databank": Version.C3,
        "crop_size": 4,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": None,
        "qa_threshold": None,
        "aggregation_level": "cell",
    }

    output_parquet = db_builds.build_databank(
        build_opt,
        str(qa_parquet),
        str(tmp_path / "out_cell_c3"),
    )
    out_df = pd.read_parquet(output_parquet)
    stacked = tifffile.imread(out_df.loc[0, "image_path"])

    expected_overlap = np.array(
        [
            [0, 0, 0, 0],
            [0, 128, 255, 128],
            [0, 128, 255, 128],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    expected_union = np.array(
        [
            [0, 0, 0, 0],
            [0, 255, 255, 255],
            [0, 255, 255, 255],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    expected_disagreement = np.array(
        [
            [0, 0, 0, 0],
            [0, 255, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert stacked.shape == (4, 4, 4)
    assert np.array_equal(stacked[0], expected_overlap)
    assert np.array_equal(stacked[1], expected_union)
    assert np.array_equal(stacked[2], expected_disagreement)
    assert np.array_equal(stacked[3], (gt[4:8, 4:8] == 1).astype(np.uint8) * 255)
