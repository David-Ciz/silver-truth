from pathlib import Path

import pandas as pd

from silver_truth.job_file_generator import generate_job_file


def test_generate_job_file_excludes_silver_truth_from_metadata(tmp_path: Path) -> None:
    parquet_path = tmp_path / "BF-C2DL-HSC_split_fold-2.parquet"
    output_dir = tmp_path / "job_files"

    df = pd.DataFrame(
        [
            {
                "campaign_number": "01",
                "CALT-US": "data/synchronized_data/BF-C2DL-HSC/CALT-US/01_RES/mask0648.tif",
                "DREX-US": "data/synchronized_data/BF-C2DL-HSC/DREX-US/01_RES/mask0648.tif",
                "SILVER-TRUTH": "data/synchronized_data/BF-C2DL-HSC/01_ST/SEG/mask0648.tif",
                "tracking_markers": "data/synchronized_data/BF-C2DL-HSC/01_GT/TRA/man_track0648.tif",
                "gt_image": "data/synchronized_data/BF-C2DL-HSC/01_GT/SEG/man_seg0648.tif",
                "source_image": "data/synchronized_data/BF-C2DL-HSC/01/t0648.tif",
                "split": "test",
            }
        ]
    )
    df.attrs["competitor_columns"] = ["CALT-US", "SILVER-TRUTH", "DREX-US"]
    df.to_parquet(parquet_path)

    generate_job_file(
        parquet_file_path=str(parquet_path),
        campaign_number="01",
        output_dir=str(output_dir),
    )

    job_file = output_dir / "BF-C2DL-HSC_01_job_file.txt"
    lines = job_file.read_text(encoding="utf-8").strip().splitlines()
    weighted_job_file = output_dir / "BF-C2DL-HSC_01_job_file_with_weights.txt"
    weighted_lines = weighted_job_file.read_text(encoding="utf-8").strip().splitlines()

    assert lines == [
        "data/synchronized_data/BF-C2DL-HSC/CALT-US/01_RES/maskTTTT.tif",
        "data/synchronized_data/BF-C2DL-HSC/DREX-US/01_RES/maskTTTT.tif",
        "data/synchronized_data/BF-C2DL-HSC/01_GT/TRA/man_trackTTTT.tif",
    ]
    assert weighted_lines == [
        "data/synchronized_data/BF-C2DL-HSC/CALT-US/01_RES/maskTTTT.tif 1",
        "data/synchronized_data/BF-C2DL-HSC/DREX-US/01_RES/maskTTTT.tif 1",
        "data/synchronized_data/BF-C2DL-HSC/01_GT/TRA/man_trackTTTT.tif",
    ]
