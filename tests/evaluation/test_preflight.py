from pathlib import Path

import pandas as pd

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    save_dataframe_to_parquet_with_metadata,
)
from silver_truth.evaluation.preflight import (
    build_split_sanity_audit,
    write_split_sanity_bundle,
)


def _write_parquets(
    tmp_path: Path, *, train_qa_rows: int, validation_qa_rows: int
) -> tuple[Path, Path]:
    whole_df = pd.DataFrame(
        [
            {
                "composite_key": "01_0000.tif",
                "campaign_number": "01",
                "time_frame": 0,
                "gt_image": "gt_01_0000.tif",
                "has_gt": True,
                "gt_cell_count": 8,
                "split": "train",
            },
            {
                "composite_key": "01_0001.tif",
                "campaign_number": "01",
                "time_frame": 1,
                "gt_image": "gt_01_0001.tif",
                "has_gt": True,
                "gt_cell_count": 7,
                "split": "train",
            },
            {
                "composite_key": "01_0002.tif",
                "campaign_number": "01",
                "time_frame": 2,
                "gt_image": "gt_01_0002.tif",
                "has_gt": True,
                "gt_cell_count": 4,
                "split": "validation",
            },
            {
                "composite_key": "02_0000.tif",
                "campaign_number": "02",
                "time_frame": 0,
                "gt_image": "gt_02_0000.tif",
                "has_gt": True,
                "gt_cell_count": 5,
                "split": "test",
            },
        ]
    )
    whole_df.attrs["split_strategy"] = "supervised_gt_frame_cells"
    whole_df.attrs["split_unit"] = "gt_cells"
    whole_df.attrs["target_val_fraction"] = 0.20

    qa_rows = []
    for idx in range(train_qa_rows):
        qa_rows.append(
            {
                "cell_id": f"train_{idx}",
                "split": "train",
                "campaign_number": "01",
                "gt_image": "gt_01_0000.tif" if idx % 2 == 0 else "gt_01_0001.tif",
                "label": idx,
            }
        )
    for idx in range(validation_qa_rows):
        qa_rows.append(
            {
                "cell_id": f"validation_{idx}",
                "split": "validation",
                "campaign_number": "01",
                "gt_image": "gt_01_0002.tif",
                "label": idx,
            }
        )
    for idx in range(3):
        qa_rows.append(
            {
                "cell_id": f"test_{idx}",
                "split": "test",
                "campaign_number": "02",
                "gt_image": "gt_02_0000.tif",
                "label": idx,
            }
        )
    qa_df = pd.DataFrame(qa_rows)

    whole_path = tmp_path / "whole.parquet"
    qa_path = tmp_path / "qa.parquet"
    save_dataframe_to_parquet_with_metadata(whole_df, str(whole_path))
    qa_df.to_parquet(qa_path)
    return whole_path, qa_path


def test_build_split_sanity_audit_passes_and_writes_bundle(tmp_path: Path) -> None:
    whole_path, qa_path = _write_parquets(
        tmp_path, train_qa_rows=5, validation_qa_rows=2
    )

    audit, sample_manifest = build_split_sanity_audit(
        dataset="BF-C2DL-HSC",
        crop_size=64,
        split_name="fold-1",
        whole_image_parquet=whole_path,
        qa_parquet=qa_path,
    )
    written = write_split_sanity_bundle(audit, sample_manifest, tmp_path / "audit")

    assert audit["scientifically_valid"] is True
    assert audit["hard_checks"]["train_qa_rows_gt_validation"] is True
    assert written["json"].exists()
    assert written["markdown"].exists()
    assert written["sample_manifest"].exists()


def test_build_split_sanity_audit_flags_validation_heavier_than_train(
    tmp_path: Path,
) -> None:
    whole_path, qa_path = _write_parquets(
        tmp_path, train_qa_rows=2, validation_qa_rows=5
    )

    audit, _ = build_split_sanity_audit(
        dataset="BF-C2DL-HSC",
        crop_size=64,
        split_name="fold-1",
        whole_image_parquet=whole_path,
        qa_parquet=qa_path,
    )

    assert audit["scientifically_valid"] is False
    assert audit["hard_checks"]["train_qa_rows_gt_validation"] is False


def test_build_split_sanity_audit_flags_qa_split_mismatch(tmp_path: Path) -> None:
    whole_path, qa_path = _write_parquets(
        tmp_path, train_qa_rows=5, validation_qa_rows=2
    )
    qa_df = pd.read_parquet(qa_path)
    qa_df.loc[qa_df["gt_image"] == "gt_01_0002.tif", "split"] = "train"
    qa_df.to_parquet(qa_path)

    audit, _ = build_split_sanity_audit(
        dataset="BF-C2DL-HSC",
        crop_size=64,
        split_name="fold-1",
        whole_image_parquet=whole_path,
        qa_parquet=qa_path,
    )

    assert audit["scientifically_valid"] is False
    assert audit["hard_checks"]["qa_split_matches_whole_image_split"] is False
    assert audit["qa_gt_split_mismatch_samples"]


def test_build_split_sanity_audit_uses_full_gt_path_for_alignment(
    tmp_path: Path,
) -> None:
    whole_df = pd.DataFrame(
        [
            {
                "composite_key": "01_1254.tif",
                "campaign_number": "01",
                "time_frame": 1254,
                "gt_image": "data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/man_seg1254.tif",
                "has_gt": True,
                "gt_cell_count": 8,
                "split": "train",
            },
            {
                "composite_key": "01_1301.tif",
                "campaign_number": "01",
                "time_frame": 1301,
                "gt_image": "data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/man_seg1301.tif",
                "has_gt": True,
                "gt_cell_count": 3,
                "split": "validation",
            },
            {
                "composite_key": "02_1254.tif",
                "campaign_number": "02",
                "time_frame": 1254,
                "gt_image": "data/synchronized_data/BF-C2DL-MuSC/02_GT/SEG/man_seg1254.tif",
                "has_gt": True,
                "gt_cell_count": 5,
                "split": "test",
            },
        ]
    )
    whole_path = tmp_path / "whole.parquet"
    save_dataframe_to_parquet_with_metadata(whole_df, str(whole_path))

    qa_df = pd.DataFrame(
        [
            {
                "cell_id": "train_1",
                "split": "train",
                "campaign_number": "01",
                "gt_image": "data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/man_seg1254.tif",
                "label": 1,
            },
            {
                "cell_id": "validation_1",
                "split": "validation",
                "campaign_number": "01",
                "gt_image": "data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/man_seg1301.tif",
                "label": 2,
            },
            {
                "cell_id": "test_1",
                "split": "test",
                "campaign_number": "02",
                "gt_image": "data/synchronized_data/BF-C2DL-MuSC/02_GT/SEG/man_seg1254.tif",
                "label": 3,
            },
        ]
    )
    qa_path = tmp_path / "qa.parquet"
    qa_df.to_parquet(qa_path)

    audit, _ = build_split_sanity_audit(
        dataset="BF-C2DL-MuSC",
        crop_size=64,
        split_name="fold-1",
        whole_image_parquet=whole_path,
        qa_parquet=qa_path,
    )

    assert audit["hard_checks"]["qa_split_matches_whole_image_split"] is True
    assert audit["qa_gt_split_mismatch_samples"] == []
