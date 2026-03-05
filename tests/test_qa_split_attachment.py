from pathlib import Path
import os
import sys

import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from silver_truth.qa.preprocessing import attach_split_to_qa_dataset


def test_attach_split_to_qa_dataset_by_gt_image(tmp_path: Path) -> None:
    dataset_df = pd.DataFrame(
        [
            {
                "campaign_number": "01",
                "gt_image": "data/synchronized_data/BF-C2DL-HSC/01_GT/SEG/man_seg0001.tif",
                "split": "train",
            },
            {
                "campaign_number": "01",
                "gt_image": "data/synchronized_data/BF-C2DL-HSC/01_GT/SEG/man_seg0002.tif",
                "split": "validation",
            },
            {
                "campaign_number": "02",
                "gt_image": "data/synchronized_data/BF-C2DL-HSC/02_GT/SEG/man_seg0001.tif",
                "split": "test",
            },
        ]
    )

    qa_base_df = pd.DataFrame(
        [
            {
                "cell_id": "c01_t0001_comp_1",
                "campaign_number": "01",
                "gt_image": "data/synchronized_data/BF-C2DL-HSC/01_GT/SEG/man_seg0001.tif",
                "stacked_path": "data/qa_crops/BF-C2DL-HSC/sz64/c01_t0001_comp_1.tif",
            },
            {
                "cell_id": "c01_t0002_comp_2",
                "campaign_number": "01",
                "gt_image": "data/synchronized_data/BF-C2DL-HSC/01_GT/SEG/man_seg0002.tif",
                "stacked_path": "data/qa_crops/BF-C2DL-HSC/sz64/c01_t0002_comp_2.tif",
            },
            {
                "cell_id": "c02_t0001_comp_1",
                "campaign_number": "02",
                "gt_image": "data/synchronized_data/BF-C2DL-HSC/02_GT/SEG/man_seg0001.tif",
                "stacked_path": "data/qa_crops/BF-C2DL-HSC/sz64/c02_t0001_comp_1.tif",
            },
        ]
    )

    split_parquet = tmp_path / "split.parquet"
    qa_base_parquet = tmp_path / "qa_base.parquet"
    qa_split_parquet = tmp_path / "qa_with_split.parquet"

    dataset_df.to_parquet(split_parquet)
    qa_base_df.to_parquet(qa_base_parquet)

    attach_split_to_qa_dataset(
        qa_base_parquet_path=str(qa_base_parquet),
        dataset_dataframe_path=str(split_parquet),
        output_parquet_path=str(qa_split_parquet),
    )

    out_df = pd.read_parquet(qa_split_parquet)

    assert len(out_df) == len(qa_base_df)
    assert out_df["split"].tolist() == ["train", "validation", "test"]
