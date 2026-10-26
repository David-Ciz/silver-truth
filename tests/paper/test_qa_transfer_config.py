from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qa_transfer import (
    collect_transfer_datasets,
    load_transfer_config,
    normalize_target_split,
)


def test_normalize_target_split_accepts_supported_aliases() -> None:
    assert normalize_target_split("1") == "fold-1"
    assert normalize_target_split("fold-2") == "fold-2"
    assert normalize_target_split("mixed") == "mixed"


def test_normalize_target_split_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        normalize_target_split("fold-3")


def test_load_transfer_config_resolves_paths_from_env(monkeypatch) -> None:
    runtime_root = Path("/tmp/qa-transfer-runtime")
    paper_runs_root = Path("/tmp/qa-transfer-paper-runs")
    monkeypatch.setenv("ABLATION_DATA_ROOT", str(runtime_root))
    monkeypatch.setenv("ABLATION_OUTPUT_DIR", str(paper_runs_root))

    config = load_transfer_config(
        Path("experiments/variants/qa_transfer_muscmixed_to_hsc.yaml"),
        "1",
    )

    assert config["target_split"] == "fold-1"
    assert config["source_dataset"] == "BF-C2DL-MuSC"
    assert config["target_dataset"] == "BF-C2DL-HSC"
    assert Path(config["source_model_template"]) == (
        paper_runs_root.resolve()
        / "qa_models/BF-C2DL-MuSC/sz256/qa_musc256_resnet18_qa_resnet18_mixed.pt"
    )
    assert Path(config["target_parquet_template"]) == (
        runtime_root.resolve()
        / "data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet"
    )
    assert collect_transfer_datasets(config) == ["BF-C2DL-MuSC", "BF-C2DL-HSC"]
