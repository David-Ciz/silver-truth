from pathlib import Path

import pandas as pd
import tifffile
import numpy as np

from silver_truth.data_processing.segmentation_stats import (
    build_oversized_cell_audit,
    collect_segmentation_object_stats,
    collect_segmentation_object_stats_from_dataframes,
    save_oversized_cell_visualizations,
    summarize_segmentation_object_stats,
)
from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    save_dataframe_to_parquet_with_metadata,
)


def test_collect_and_summarize_segmentation_object_stats(tmp_path: Path) -> None:
    seq1 = tmp_path / "01_GT" / "SEG"
    seq2 = tmp_path / "02_GT" / "SEG"
    seq1.mkdir(parents=True)
    seq2.mkdir(parents=True)

    img1 = np.zeros((8, 8), dtype=np.uint16)
    img1[1:3, 1:3] = 1
    img1[4:7, 4:6] = 2

    img2 = np.zeros((8, 8), dtype=np.uint16)
    img2[0:5, 0:2] = 1

    tifffile.imwrite(seq1 / "man_seg0001.tif", img1)
    tifffile.imwrite(seq2 / "man_seg0002.tif", img2)

    stats_df = collect_segmentation_object_stats([seq1, seq2])

    assert len(stats_df) == 3
    assert set(stats_df["area_px"].tolist()) == {4, 6, 10}
    assert set(stats_df["bbox_height_px"].tolist()) == {2, 3, 5}
    assert set(stats_df["bbox_width_px"].tolist()) == {2}
    assert set(stats_df["bbox_aspect_ratio_wh"].tolist()) == {1.0, 2 / 3, 0.4}
    assert set(stats_df["bbox_elongation_ratio"].tolist()) == {1.0, 1.5, 2.5}

    summary = summarize_segmentation_object_stats(
        stats_df,
        crop_sizes=(2, 4),
        rect_sizes=((2, 4), (4, 2)),
    )

    assert summary["crop_sizes"] == [2, 4]
    assert summary["rect_sizes"] == [
        {"height": 2, "width": 4},
        {"height": 4, "width": 2},
    ]
    assert summary["overall"]["n_files"] == 2
    assert summary["overall"]["n_cells"] == 3
    assert summary["overall"]["max_area_px"] == 10
    assert summary["overall"]["max_bbox_dim_px"] == 5
    assert summary["overall"]["over_bbox_count_sz2"] == 2
    assert summary["overall"]["over_bbox_count_sz4"] == 1
    assert summary["overall"]["fit_bbox_rate_sz2"] == 1 / 3
    assert summary["overall"]["fit_bbox_rate_sz4"] == 2 / 3
    assert summary["overall"]["fit_bbox_rate_rect2x4"] == 1 / 3
    assert summary["overall"]["over_bbox_count_rect2x4"] == 2
    assert summary["overall"]["fit_bbox_rate_rect2x4_swappable"] == 2 / 3
    assert summary["overall"]["over_bbox_count_rect2x4_swappable"] == 1
    assert summary["overall"]["fit_bbox_rate_rect4x2"] == 2 / 3
    assert summary["overall"]["over_bbox_count_rect4x2"] == 1
    assert summary["overall"]["fit_bbox_rate_rect4x2_swappable"] == 2 / 3
    assert summary["overall"]["over_bbox_count_rect4x2_swappable"] == 1
    assert np.isclose(summary["overall"]["mean_bbox_aspect_ratio_wh"], (1.0 + (2 / 3) + 0.4) / 3)
    assert summary["overall"]["median_bbox_aspect_ratio_wh"] == 2 / 3
    assert summary["overall"]["mean_bbox_elongation_ratio"] == (1.0 + 1.5 + 2.5) / 3
    assert summary["overall"]["max_bbox_elongation_ratio"] == 2.5
    assert summary["overall"]["wide_bbox_rate"] == 0.0
    assert summary["overall"]["tall_bbox_rate"] == 2 / 3
    assert summary["overall"]["square_bbox_rate"] == 1 / 3


def test_collect_from_parquet_and_visualize_oversized_cells(tmp_path: Path) -> None:
    source_path = tmp_path / "data" / "synchronized_data" / "BF-C2DL-MuSC" / "01" / "t0001.tif"
    gt_path = tmp_path / "data" / "synchronized_data" / "BF-C2DL-MuSC" / "01_GT" / "SEG" / "man_seg0001.tif"
    source_path.parent.mkdir(parents=True)
    gt_path.parent.mkdir(parents=True)

    source = np.arange(100, dtype=np.uint8).reshape(10, 10)
    gt = np.zeros((10, 10), dtype=np.uint16)
    gt[1:9, 3:6] = 1

    tifffile.imwrite(source_path, source)
    tifffile.imwrite(gt_path, gt)

    parquet_path = tmp_path / "data" / "dataframes" / "BF-C2DL-MuSC" / "whole_image" / "sample.parquet"
    df = pd.DataFrame(
        [
            {
                "composite_key": "01_0001.tif",
                "campaign_number": "01",
                "source_image": "data/synchronized_data/BF-C2DL-MuSC/01/t0001.tif",
                "gt_image": "data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/man_seg0001.tif",
            }
        ]
    )
    save_dataframe_to_parquet_with_metadata(df, str(parquet_path))

    stats_df = collect_segmentation_object_stats_from_dataframes([parquet_path])

    assert len(stats_df) == 1
    row = stats_df.iloc[0]
    assert row["bbox_height_px"] == 8
    assert row["bbox_width_px"] == 3
    assert row["source_image"].endswith("t0001.tif")
    assert row["gt_image"].endswith("man_seg0001.tif")

    audit_df = build_oversized_cell_audit(stats_df, crop_size=6)

    assert len(audit_df) == 1
    audit_row = audit_df.iloc[0]
    assert audit_row["oversize_reason"] == "height"
    assert audit_row["overflow_height_px"] == 2
    assert audit_row["overflow_width_px"] == 0

    output_dir = tmp_path / "audit_pngs"
    created_files = save_oversized_cell_visualizations(
        audit_df,
        output_dir=output_dir,
        crop_size=6,
        max_visualizations=5,
        context_pad=4,
    )

    assert len(created_files) == 1
    assert Path(created_files[0]).exists()
