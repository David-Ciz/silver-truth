from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from silver_truth.ensemble import databanks_builds as db_builds
from silver_truth.ensemble.datasets import Version

tifffile = pytest.importorskip("tifffile")


def _write_full_stack(path: Path, mask: np.ndarray) -> None:
    raw = np.zeros(mask.shape, dtype=np.uint8)
    stacked = np.stack([raw, mask.astype(np.uint8)], axis=0)
    tifffile.imwrite(path, stacked)


def _write_crop_stack(path: Path, mask: np.ndarray) -> None:
    raw = np.zeros(mask.shape, dtype=np.uint8)
    gt = np.zeros(mask.shape, dtype=np.uint8)
    tra = np.zeros(mask.shape, dtype=np.uint8)
    stacked = np.stack([raw, mask.astype(np.uint8), gt, tra], axis=0)
    tifffile.imwrite(path, stacked)


def _base_build_opt() -> dict:
    return {
        "name": "BF-C2DL-HSC",
        "version": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "aggregation_level": "image",
    }


def test_build_databank_image_level_combines_all_cell_masks(
    tmp_path: Path, monkeypatch
) -> None:
    gt = np.zeros((6, 6), dtype=np.uint8)
    gt[1:3, 1:3] = 1
    gt[3:5, 3:5] = 2
    gt_path = tmp_path / "gt.tif"
    tifffile.imwrite(gt_path, gt)

    mask_a1 = np.zeros_like(gt, dtype=np.uint8)
    mask_a1[1:3, 1:3] = 255
    mask_b1 = np.zeros_like(gt, dtype=np.uint8)
    mask_b1[1:3, 2:4] = 255

    mask_a2 = np.zeros_like(gt, dtype=np.uint8)
    mask_a2[3:5, 3:5] = 255
    mask_b2 = np.zeros_like(gt, dtype=np.uint8)
    mask_b2[2:4, 3:5] = 255

    paths = {
        "a1": tmp_path / "a1.tif",
        "b1": tmp_path / "b1.tif",
        "a2": tmp_path / "a2.tif",
        "b2": tmp_path / "b2.tif",
    }
    _write_full_stack(paths["a1"], mask_a1)
    _write_full_stack(paths["b1"], mask_b1)
    _write_full_stack(paths["a2"], mask_a2)
    _write_full_stack(paths["b2"], mask_b2)

    qa_df = pd.DataFrame(
        [
            {
                "gt_image": str(gt_path),
                "label": 1,
                "stacked_path": str(paths["a1"]),
                "split": "train",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "crop_size": None,
            },
            {
                "gt_image": str(gt_path),
                "label": 1,
                "stacked_path": str(paths["b1"]),
                "split": "train",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "crop_size": None,
            },
            {
                "gt_image": str(gt_path),
                "label": 2,
                "stacked_path": str(paths["a2"]),
                "split": "train",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "crop_size": None,
            },
            {
                "gt_image": str(gt_path),
                "label": 2,
                "stacked_path": str(paths["b2"]),
                "split": "train",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "crop_size": None,
            },
        ]
    )
    qa_parquet = tmp_path / "qa.parquet"
    qa_df.to_parquet(qa_parquet)

    monkeypatch.setattr(db_builds.ext, "compress_images", lambda *args, **kwargs: None)

    build_opt = _base_build_opt()
    build_opt.update({"qa": None, "qa_threshold": None})

    output_parquet = db_builds.build_databank(
        build_opt,
        str(qa_parquet),
        str(tmp_path / "out"),
    )

    out_df = pd.read_parquet(output_parquet)
    assert len(out_df) == 1
    assert out_df.loc[0, "split"] == "train"
    assert pd.isna(out_df.loc[0, "crop_size"])

    stacked = tifffile.imread(out_df.loc[0, "image_path"])
    expected_votes = np.clip(
        ((mask_a1.astype(np.int32) + mask_b1.astype(np.int32)) // 2)
        + ((mask_a2.astype(np.int32) + mask_b2.astype(np.int32)) // 2),
        0,
        255,
    ).astype(np.uint8)
    expected_gt = (gt > 0).astype(np.uint8) * 255

    assert stacked.shape == (3, 6, 6)
    assert np.array_equal(stacked[0], expected_votes)
    assert np.array_equal(stacked[1], expected_gt)


def test_build_databank_image_level_respects_qa_threshold(
    tmp_path: Path, monkeypatch
) -> None:
    gt = np.zeros((6, 6), dtype=np.uint8)
    gt[1:3, 1:3] = 1
    gt[3:5, 3:5] = 2
    gt_path = tmp_path / "gt.tif"
    tifffile.imwrite(gt_path, gt)

    mask_a1 = np.zeros_like(gt, dtype=np.uint8)
    mask_a1[1:3, 1:3] = 255
    mask_b1 = np.zeros_like(gt, dtype=np.uint8)
    mask_b1[1:3, 2:4] = 255

    mask_a2 = np.zeros_like(gt, dtype=np.uint8)
    mask_a2[3:5, 3:5] = 255
    mask_b2 = np.zeros_like(gt, dtype=np.uint8)
    mask_b2[2:4, 3:5] = 255

    paths = {
        "a1": tmp_path / "a1.tif",
        "b1": tmp_path / "b1.tif",
        "a2": tmp_path / "a2.tif",
        "b2": tmp_path / "b2.tif",
    }
    _write_full_stack(paths["a1"], mask_a1)
    _write_full_stack(paths["b1"], mask_b1)
    _write_full_stack(paths["a2"], mask_a2)
    _write_full_stack(paths["b2"], mask_b2)

    qa_df = pd.DataFrame(
        [
            {
                "gt_image": str(gt_path),
                "label": 1,
                "stacked_path": str(paths["a1"]),
                "split": "train",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "crop_size": None,
                "qa_score": 0.95,
            },
            {
                "gt_image": str(gt_path),
                "label": 1,
                "stacked_path": str(paths["b1"]),
                "split": "train",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "crop_size": None,
                "qa_score": 0.20,
            },
            {
                "gt_image": str(gt_path),
                "label": 2,
                "stacked_path": str(paths["a2"]),
                "split": "train",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "crop_size": None,
                "qa_score": 0.10,
            },
            {
                "gt_image": str(gt_path),
                "label": 2,
                "stacked_path": str(paths["b2"]),
                "split": "train",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "crop_size": None,
                "qa_score": 0.90,
            },
        ]
    )
    qa_parquet = tmp_path / "qa.parquet"
    qa_df.to_parquet(qa_parquet)

    monkeypatch.setattr(db_builds.ext, "compress_images", lambda *args, **kwargs: None)

    build_opt = _base_build_opt()
    build_opt.update({"qa": "qa_score", "qa_threshold": 0.8})

    output_parquet = db_builds.build_databank(
        build_opt,
        str(qa_parquet),
        str(tmp_path / "out"),
    )
    out_df = pd.read_parquet(output_parquet)
    stacked = tifffile.imread(out_df.loc[0, "image_path"])

    # label 1 keeps competitor A, label 2 keeps competitor B.
    expected_votes = np.clip(
        mask_a1.astype(np.int32) + mask_b2.astype(np.int32), 0, 255
    ).astype(np.uint8)
    assert np.array_equal(stacked[0], expected_votes)


def test_build_databank_cell_level_includes_reconstruction_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    gt = np.zeros((12, 12), dtype=np.uint8)
    gt[4:8, 4:8] = 1
    gt_path = tmp_path / "gt_cell.tif"
    tifffile.imwrite(gt_path, gt)

    mask_a = np.zeros((4, 4), dtype=np.uint8)
    mask_a[1:3, 1:3] = 255
    mask_b = np.zeros((4, 4), dtype=np.uint8)
    mask_b[1:3, 1:3] = 255

    stack_a = tmp_path / "cell_a.tif"
    stack_b = tmp_path / "cell_b.tif"
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
                "centering": "competitor_consensus",
                "center_agreement_count": 2,
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
                "centering": "competitor_consensus",
                "center_agreement_count": 2,
            },
        ]
    )
    qa_parquet = tmp_path / "qa_cell.parquet"
    qa_df.to_parquet(qa_parquet)

    monkeypatch.setattr(db_builds.ext, "compress_images", lambda *args, **kwargs: None)

    build_opt = {
        "name": "BF-C2DL-HSC",
        "version": Version.C1,
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
        str(tmp_path / "out_cell"),
    )
    out_df = pd.read_parquet(output_parquet)
    assert len(out_df) == 1

    row = out_df.iloc[0]
    assert row["campaign_number"] == "01"
    assert row["original_image_key"] == "t0001"
    assert row["qa_centering"] == "competitor_consensus"
    assert row["qa_center_agreement_count"] == 2
    assert row["recon_crop_y_end"] - row["recon_crop_y_start"] == row["crop_size"]
    assert row["recon_crop_x_end"] - row["recon_crop_x_start"] == row["crop_size"]
