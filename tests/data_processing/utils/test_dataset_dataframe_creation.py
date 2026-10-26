import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    REFERENCE_COLUMNS_ATTR,
    SILVER_TRUTH_COLUMN,
    create_dataset_dataframe_logic,
    load_dataframe_from_parquet_with_metadata,
    process_dataset_directory,
    save_dataframe_to_parquet_with_metadata,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_process_dataset_directory_includes_silver_truth(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "BF-C2DL-HSC"

    _touch(dataset_dir / "01" / "t0001.tif")
    _touch(dataset_dir / "02" / "t0001.tif")
    _touch(dataset_dir / "01_GT" / "SEG" / "man_seg0001.tif")
    _touch(dataset_dir / "01_GT" / "TRA" / "man_track0001.tif")
    _touch(dataset_dir / "02_GT" / "SEG" / "man_seg0001.tif")
    _touch(dataset_dir / "02_GT" / "TRA" / "man_track0001.tif")
    _touch(dataset_dir / "01_ST" / "SEG" / "mask0001.tif")
    _touch(dataset_dir / "02_ST" / "SEG" / "mask0001.tif")
    _touch(dataset_dir / "CALT-US" / "01_RES" / "mask0001.tif")
    _touch(dataset_dir / "CALT-US" / "02_RES" / "mask0001.tif")

    dataset_info, competitor_columns = process_dataset_directory(dataset_dir)

    assert SILVER_TRUTH_COLUMN not in competitor_columns
    assert "CALT-US" in competitor_columns
    assert dataset_info["01_0001.tif"][SILVER_TRUTH_COLUMN].endswith(
        "01_ST/SEG/mask0001.tif"
    )
    assert dataset_info["02_0001.tif"][SILVER_TRUTH_COLUMN].endswith(
        "02_ST/SEG/mask0001.tif"
    )


def test_save_dataframe_creates_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "dir" / "dataset.parquet"
    df = pd.DataFrame({"value": [1, 2]})
    df.attrs["competitor_columns"] = ["CALT-US"]

    save_dataframe_to_parquet_with_metadata(df, str(output_path))

    assert output_path.exists()
    loaded_df = load_dataframe_from_parquet_with_metadata(str(output_path))
    assert loaded_df["value"].tolist() == [1, 2]
    assert loaded_df.attrs["competitor_columns"] == ["CALT-US"]


def test_reference_columns_metadata_is_preserved(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.parquet"
    df = pd.DataFrame({"value": [1], SILVER_TRUTH_COLUMN: ["data/01_ST/SEG/mask0001.tif"]})
    df.attrs["competitor_columns"] = ["CALT-US"]
    df.attrs[REFERENCE_COLUMNS_ATTR] = [SILVER_TRUTH_COLUMN]

    save_dataframe_to_parquet_with_metadata(df, str(output_path))

    loaded_df = load_dataframe_from_parquet_with_metadata(str(output_path))
    assert loaded_df.attrs[REFERENCE_COLUMNS_ATTR] == [SILVER_TRUTH_COLUMN]


def test_create_dataset_dataframe_logic_writes_fold_split_audit(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "synchronized_data" / "BF-C2DL-HSC"
    output_path = (
        tmp_path
        / "data"
        / "dataframes"
        / "BF-C2DL-HSC"
        / "whole_image"
        / "BF-C2DL-HSC_split_fold-1.parquet"
    )

    def _write_image(path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, array.astype(np.uint16))

    raw = np.zeros((4, 4), dtype=np.uint16)
    one_cell = np.zeros((4, 4), dtype=np.uint16)
    one_cell[1:3, 1:3] = 1
    two_cells = np.zeros((4, 4), dtype=np.uint16)
    two_cells[0:2, 0:2] = 1
    two_cells[2:4, 2:4] = 2

    for frame in range(4):
        _write_image(dataset_dir / "01" / f"t{frame:04d}.tif", raw)
        _write_image(dataset_dir / "02" / f"t{frame:04d}.tif", raw)

    _write_image(dataset_dir / "01_GT" / "SEG" / "man_seg0000.tif", one_cell)
    _write_image(dataset_dir / "01_GT" / "SEG" / "man_seg0002.tif", two_cells)
    _write_image(dataset_dir / "01_GT" / "TRA" / "man_track0000.tif", one_cell)
    _write_image(dataset_dir / "01_GT" / "TRA" / "man_track0002.tif", two_cells)
    _write_image(dataset_dir / "02_GT" / "SEG" / "man_seg0001.tif", one_cell)
    _write_image(dataset_dir / "02_GT" / "TRA" / "man_track0001.tif", one_cell)

    create_dataset_dataframe_logic(
        synchronized_dataset_dir=dataset_dir,
        output_path=output_path,
        split_mode="fold-1",
        split_ratios="80,20",
    )

    audit_json = (
        tmp_path
        / "data"
        / "dataframes"
        / "BF-C2DL-HSC"
        / "split_audits"
        / "BF-C2DL-HSC_fold-1_split_audit.json"
    )
    audit_md = audit_json.with_suffix(".md")
    assert output_path.exists()
    assert audit_json.exists()
    assert audit_md.exists()

    loaded_df = load_dataframe_from_parquet_with_metadata(str(output_path))
    audit = json.loads(audit_json.read_text(encoding="utf-8"))

    assert loaded_df.attrs["split_strategy"] == "supervised_gt_frame_cells"
    assert loaded_df.attrs["split_unit"] == "gt_cells"
    assert loaded_df.attrs["validation_selection_mode"] in {
        "contiguous_gt_block",
        "subset_sum_fallback",
    }
    assert audit["dataset"] == "BF-C2DL-HSC"
    assert audit["fold"] == "fold-1"
