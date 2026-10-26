from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silver_truth.qa import tabular as qa_tabular


tifffile = pytest.importorskip("tifffile")


def _write_stacked_tiff(path: Path, raw_value: int, mask_size: int) -> None:
    stacked = np.zeros((4, 8, 8), dtype=np.uint8)
    stacked[0] = raw_value
    stacked[1, :mask_size, :mask_size] = 255
    tifffile.imwrite(path, stacked)


def test_tabular_backend_trains_and_writes_excel(tmp_path: Path) -> None:
    rows = []
    split_order = ["train", "train", "train", "validation", "test"]
    for idx, split in enumerate(split_order):
        image_path = tmp_path / f"sample_{idx}.tif"
        _write_stacked_tiff(image_path, raw_value=20 + idx, mask_size=2 + (idx % 3))
        rows.append(
            {
                "stacked_path": str(image_path),
                "jaccard_score": 0.2 + (0.1 * idx),
                "split": split,
                "cell_id": f"cell-{idx}",
                "center_agreement_count": 1 + idx,
                "crop_size": 8,
            }
        )

    parquet_path = tmp_path / "qa.parquet"
    model_path = tmp_path / "qa_tabular.joblib"
    excel_path = tmp_path / "qa_tabular.xlsx"
    pd.DataFrame(rows).to_parquet(parquet_path)

    qa_tabular.train(
        parquet_file=parquet_path,
        metadata_features="default",
        output_model=str(model_path),
        output_excel=str(excel_path),
        model_type="hist_gradient_boosting",
        seed=7,
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment="qa-tabular-test",
        mlflow_run_name="tabular-test",
    )

    assert model_path.exists()
    assert excel_path.exists()

    workbook = pd.ExcelFile(excel_path)
    assert set(workbook.sheet_names) >= {"train", "validation", "test"}

    test_df = pd.read_excel(workbook, sheet_name="test")
    assert "Predicted Jaccard index" in test_df.columns
    assert test_df["Predicted Jaccard index"].between(0.0, 1.0).all()
