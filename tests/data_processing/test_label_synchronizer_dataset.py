from pathlib import Path

import silver_truth.data_processing.label_synchronizer as label_synchronizer


def test_verify_dataset_synchronization_logic_ignores_empty_results(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "BF-C2DL-MuSC"
    (dataset_dir / "01_GT" / "SEG").mkdir(parents=True)
    (dataset_dir / "02_GT" / "SEG").mkdir(parents=True)
    (dataset_dir / "CALT-US" / "01_RES").mkdir(parents=True)
    (dataset_dir / "CALT-US" / "02_RES").mkdir(parents=True)

    track_01 = tmp_path / "01_TRA"
    track_02 = tmp_path / "02_TRA"
    track_01.mkdir()
    track_02.mkdir()

    monkeypatch.setattr(
        label_synchronizer,
        "verify_folder_synchronization_logic",
        lambda label_folder, tracking_folder: [],
    )

    result = label_synchronizer.verify_dataset_synchronization_logic(
        dataset_dir, track_01, track_02
    )

    assert result == set()


def test_verify_dataset_synchronization_logic_reports_only_bad_split(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "BF-C2DL-MuSC"
    seg_01 = dataset_dir / "01_GT" / "SEG"
    seg_02 = dataset_dir / "02_GT" / "SEG"
    comp_01 = dataset_dir / "CALT-US" / "01_RES"
    comp_02 = dataset_dir / "CALT-US" / "02_RES"
    seg_01.mkdir(parents=True)
    seg_02.mkdir(parents=True)
    comp_01.mkdir(parents=True)
    comp_02.mkdir(parents=True)

    track_01 = tmp_path / "01_TRA"
    track_02 = tmp_path / "02_TRA"
    track_01.mkdir()
    track_02.mkdir()

    bad_by_folder = {
        seg_01: [],
        comp_01: ["0005"],
        seg_02: [],
        comp_02: [],
    }

    def fake_verify(label_folder, tracking_folder):
        return bad_by_folder[Path(label_folder)]

    monkeypatch.setattr(
        label_synchronizer,
        "verify_folder_synchronization_logic",
        fake_verify,
    )

    result = label_synchronizer.verify_dataset_synchronization_logic(
        dataset_dir, track_01, track_02
    )

    assert result == {"01"}
