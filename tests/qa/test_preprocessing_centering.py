from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from silver_truth.qa.preprocessing import create_qa_dataset


tifffile = pytest.importorskip("tifffile")


def _make_source_tree(root: Path) -> tuple[Path, Path, Path]:
    dataset_root = root / "synchronized_data" / "TinyDataset"
    gt_dir = dataset_root / "01_GT" / "SEG"
    raw_dir = dataset_root / "01"
    gt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return dataset_root, gt_dir, raw_dir


def test_create_qa_dataset_competitor_alias_uses_consensus_center(
    tmp_path: Path,
) -> None:
    dataset_root, gt_dir, raw_dir = _make_source_tree(tmp_path)

    raw = np.zeros((8, 8), dtype=np.uint8)
    raw_path = raw_dir / "t001.tif"
    tifffile.imwrite(raw_path, raw)

    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[2:5, 2:5] = 1
    gt_path = gt_dir / "man_seg001.tif"
    tifffile.imwrite(gt_path, gt)

    seg_a = np.zeros((8, 8), dtype=np.uint8)
    seg_a[1:3, 1:3] = 1
    seg_b = np.zeros((8, 8), dtype=np.uint8)
    seg_b[4:6, 4:6] = 1
    seg_a_path = dataset_root / "seg_a.tif"
    seg_b_path = dataset_root / "seg_b.tif"
    tifffile.imwrite(seg_a_path, seg_a)
    tifffile.imwrite(seg_b_path, seg_b)

    df = pd.DataFrame(
        [
            {
                "composite_key": "01_001.tif",
                "campaign_number": "01",
                "gt_image": str(gt_path),
                "comp_a": str(seg_a_path),
                "comp_b": str(seg_b_path),
            }
        ]
    )
    parquet_path = tmp_path / "dataset.parquet"
    df.to_parquet(parquet_path)

    out_dir = tmp_path / "qa_out"
    out_parquet = tmp_path / "qa.parquet"
    create_qa_dataset(
        dataset_dataframe_path=str(parquet_path),
        output_dir=str(out_dir),
        output_parquet_path=str(out_parquet),
        crop=True,
        crop_size=4,
        centering="competitor",
    )

    out_df = pd.read_parquet(out_parquet)
    assert len(out_df) == 2
    assert set(out_df["centering"].unique()) == {"competitor_consensus"}
    assert out_df["original_center_y"].nunique() == 1
    assert out_df["original_center_x"].nunique() == 1
    assert set(out_df["center_agreement_count"].unique()) == {2}


def test_create_qa_dataset_competitor_individual_keeps_per_competitor_centers(
    tmp_path: Path,
) -> None:
    dataset_root, gt_dir, raw_dir = _make_source_tree(tmp_path)

    raw = np.zeros((8, 8), dtype=np.uint8)
    raw_path = raw_dir / "t001.tif"
    tifffile.imwrite(raw_path, raw)

    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[2:5, 2:5] = 1
    gt_path = gt_dir / "man_seg001.tif"
    tifffile.imwrite(gt_path, gt)

    seg_a = np.zeros((8, 8), dtype=np.uint8)
    seg_a[1:3, 1:3] = 1
    seg_b = np.zeros((8, 8), dtype=np.uint8)
    seg_b[4:6, 4:6] = 1
    seg_a_path = dataset_root / "seg_a.tif"
    seg_b_path = dataset_root / "seg_b.tif"
    tifffile.imwrite(seg_a_path, seg_a)
    tifffile.imwrite(seg_b_path, seg_b)

    df = pd.DataFrame(
        [
            {
                "composite_key": "01_001.tif",
                "campaign_number": "01",
                "gt_image": str(gt_path),
                "comp_a": str(seg_a_path),
                "comp_b": str(seg_b_path),
            }
        ]
    )
    parquet_path = tmp_path / "dataset.parquet"
    df.to_parquet(parquet_path)

    out_dir = tmp_path / "qa_out"
    out_parquet = tmp_path / "qa_individual.parquet"
    create_qa_dataset(
        dataset_dataframe_path=str(parquet_path),
        output_dir=str(out_dir),
        output_parquet_path=str(out_parquet),
        crop=True,
        crop_size=4,
        centering="competitor_individual",
    )

    out_df = pd.read_parquet(out_parquet)
    assert len(out_df) == 2
    assert set(out_df["centering"].unique()) == {"competitor_individual"}
    assert out_df["original_center_y"].nunique() == 2
    assert out_df["original_center_x"].nunique() == 2
