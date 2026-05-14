import sys
from click.testing import CliRunner
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from silver_truth.cli.ensemble import build_databank
from silver_truth.dataset_registry import dataset_id, infer_dataset_name_from_text
from scripts.run_ablation import _databank_parquet


def test_legacy_datasets_keep_existing_databank_ids() -> None:
    assert (
        _databank_parquet("/tmp/databank", "BF-C2DL-HSC")
        == "/tmp/databank/C1_ds1-42-7015_QA--.parquet"
    )


def test_new_ctc_datasets_get_deterministic_databank_ids() -> None:
    assert dataset_id("Fluo-C2DL-MSC") == "fluo-c2dl-msc"
    assert dataset_id("Fluo-N2DH-SIM+") == "fluo-n2dh-simplus"
    assert (
        _databank_parquet("/tmp/databank", "Fluo-C2DL-MSC")
        == "/tmp/databank/C1_fluo-c2dl-msc-42-7015_QA--.parquet"
    )


def test_build_databank_cli_accepts_fluo_c2dl_msc_dataset_name() -> None:
    result = CliRunner().invoke(
        build_databank,
        [
            "--dataset-name",
            "Fluo-C2DL-MSC",
            "--qa-parquet-path",
            "/does/not/exist.parquet",
        ],
    )

    assert "Invalid value for '--dataset-name'" not in result.output
    assert "File '/does/not/exist.parquet' does not exist" in result.output


def test_dataset_name_inference_handles_future_ctc_datasets() -> None:
    assert (
        infer_dataset_name_from_text(["/runs/Fluo-C3DH-A549-SIM/sz1024/output"])
        == "Fluo-C3DH-A549-SIM"
    )
    assert (
        infer_dataset_name_from_text(["/runs/Fluo-N2DH-SIM+/sz512/output"])
        == "Fluo-N2DH-SIM+"
    )
