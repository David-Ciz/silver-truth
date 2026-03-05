from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from silver_truth.fusion import crops_experiment as ce


mlflow = pytest.importorskip("mlflow")
tifffile = pytest.importorskip("tifffile")


def _write_stack(path: Path, seg: np.ndarray) -> None:
    raw = np.full(seg.shape, 100, dtype=np.uint8)
    stack = np.stack([raw, seg.astype(np.uint8)], axis=0)
    tifffile.imwrite(path, stack)


def _make_qa_df(
    stack_a: Path,
    stack_b: Path,
    gt_image: Path,
    split: str = "train",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 5,
                "competitor": "Comp A",
                "stacked_path": str(stack_a),
                "split": split,
                "gt_image": str(gt_image),
                "crop_y_start": 2,
                "crop_y_end": 6,
                "crop_x_start": 3,
                "crop_x_end": 7,
            },
            {
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 5,
                "competitor": "KTH-SE (5)",
                "stacked_path": str(stack_b),
                "split": split,
                "gt_image": str(gt_image),
                "crop_y_start": 2,
                "crop_y_end": 6,
                "crop_x_start": 3,
                "crop_x_end": 7,
            },
        ]
    )


def test_prepare_fusion_job_reconstructs_gt_and_sanitizes_competitor_dirs(
    tmp_path: Path,
) -> None:
    seg_a = np.zeros((4, 4), dtype=np.uint8)
    seg_a[1:3, 1:3] = 1
    seg_b = np.zeros((4, 4), dtype=np.uint8)
    seg_b[0:2, 0:2] = 1

    stack_a = tmp_path / "comp_a.tif"
    stack_b = tmp_path / "comp_b.tif"
    _write_stack(stack_a, seg_a)
    _write_stack(stack_b, seg_b)

    gt_full = np.zeros((10, 10), dtype=np.uint8)
    gt_full[3:5, 4:6] = 5
    gt_image = tmp_path / "gt.tif"
    tifffile.imwrite(gt_image, gt_full)

    qa_path = tmp_path / "qa.parquet"
    df = _make_qa_df(stack_a=stack_a, stack_b=stack_b, gt_image=gt_image)
    df.to_parquet(qa_path, index=False)

    cell_groups, cell_splits, cell_metadata = ce.build_cell_groups(df, qa_path)
    competitors = sorted(df["competitor"].unique().tolist())
    competitor_dir_names = ce.build_competitor_dir_names(competitors)

    mapping = ce.prepare_fusion_job(
        job_dir=tmp_path / "job",
        cell_groups=cell_groups,
        competitors=competitors,
        cell_metadata=cell_metadata,
        competitor_dir_names=competitor_dir_names,
    )

    assert len(mapping) == 1
    assert list(cell_splits.values()) == ["train"]
    assert competitor_dir_names["KTH-SE (5)"] == "KTH-SE_5"

    gt_mask = tifffile.imread(tmp_path / "job" / "GT" / "mask0000.tif")
    tra_mask = tifffile.imread(tmp_path / "job" / "TRA" / "mask0000.tif")
    assert int((gt_mask > 0).sum()) > 0
    assert np.array_equal(tra_mask, gt_mask)


def test_evaluate_model_results_nonempty_gt_empty_fused_is_zero(tmp_path: Path) -> None:
    fusion_out = tmp_path / "fused"
    gt_dir = tmp_path / "gt"
    fusion_out.mkdir()
    gt_dir.mkdir()

    tifffile.imwrite(fusion_out / "fused_0000.tif", np.zeros((4, 4), dtype=np.uint8))
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[1:3, 1:3] = 1
    tifffile.imwrite(gt_dir / "mask0000.tif", gt)

    mapping = {0: ("01", "t0001", 5)}
    splits = {("01", "t0001", 5): "test"}
    results = ce.evaluate_model_results(
        fusion_out_dir=fusion_out,
        gt_dir=gt_dir,
        mapping=mapping,
        cell_splits=splits,
        model="THRESHOLD_FLAT",
    )

    assert len(results) == 1
    assert float(results.loc[0, "jaccard"]) == 0.0
    assert float(results.loc[0, "f1"]) == 0.0
    assert results.loc[0, "split"] == "test"


def test_inspect_fused_outputs_reports_counts_and_sample(tmp_path: Path) -> None:
    out = tmp_path / "thr_0.60"
    out.mkdir()

    tifffile.imwrite(out / "fused_0000.tif", np.ones((4, 4), dtype=np.uint8))
    tifffile.imwrite(out / "fused_0006.tif", np.zeros((4, 4), dtype=np.uint8))

    mapping = {
        0: ("01", "t0001", 1),
        1: ("01", "t0002", 1),
        6: ("01", "t0006", 1),
    }
    metrics = ce.inspect_fused_outputs(out, mapping)

    assert metrics["expected_outputs"] == 3.0
    assert metrics["present_outputs"] == 2.0
    assert metrics["missing_fused_outputs"] == 1.0
    assert metrics["nonempty_fused_outputs"] == 1.0
    assert metrics["sample_0006_exists"] == 1.0
    assert metrics["sample_0006_nonempty"] == 0.0


def test_run_crops_fusion_experiment_skip_fusion_writes_split_aware_parquets(
    tmp_path: Path, monkeypatch
) -> None:
    seg_a = np.zeros((4, 4), dtype=np.uint8)
    seg_a[1:3, 1:3] = 1
    seg_b = np.zeros((4, 4), dtype=np.uint8)
    seg_b[0:2, 0:2] = 1

    stack_a = tmp_path / "comp_a.tif"
    stack_b = tmp_path / "comp_b.tif"
    _write_stack(stack_a, seg_a)
    _write_stack(stack_b, seg_b)

    gt_full = np.zeros((10, 10), dtype=np.uint8)
    gt_full[3:5, 4:6] = 5
    gt_image = tmp_path / "gt.tif"
    tifffile.imwrite(gt_image, gt_full)

    qa_path = tmp_path / "qa.parquet"
    df = _make_qa_df(
        stack_a=stack_a, stack_b=stack_b, gt_image=gt_image, split="validation"
    )
    df.to_parquet(qa_path, index=False)

    output_dir = tmp_path / "out"
    model_dir = output_dir / "threshold_flat"
    model_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(model_dir / "fused_0000.tif", np.zeros((4, 4), dtype=np.uint8))

    fake_jar = tmp_path / "fake.jar"
    fake_jar.write_text("jar", encoding="utf-8")
    monkeypatch.setattr(ce, "DEFAULT_JAR_PATH", fake_jar)

    result = ce.run_crops_fusion_experiment(
        qa_parquet=qa_path,
        output_dir=output_dir,
        models=("THRESHOLD_FLAT",),
        skip_fusion=True,
        chunk_size=0,
        mlflow_experiment="test-fusion-crops",
        mlflow_tracking_path=tmp_path / "mlruns",
    )

    assert result["models_run"] == ["THRESHOLD_FLAT"]
    assert Path(result["leaderboard_path"]).exists()

    eval_parquet = model_dir / f"{qa_path.stem}_threshold_flat_cell_eval.parquet"
    with_fused_parquet = model_dir / f"{qa_path.stem}_threshold_flat_with_fused.parquet"
    assert eval_parquet.exists()
    assert with_fused_parquet.exists()

    eval_df = pd.read_parquet(eval_parquet)
    assert len(eval_df) == 1
    assert eval_df.loc[0, "split"] == "validation"
    assert pd.notna(eval_df.loc[0, "threshold_flat"])

    summary_df = pd.read_csv(
        output_dir / "threshold_flat" / "threshold_flat_summary.csv"
    )
    assert float(summary_df.loc[0, "mean_jaccard"]) == 0.0
    assert float(summary_df.loc[0, "mean_f1"]) == 0.0
    assert float(summary_df.loc[0, "eval_overall_mean_jaccard"]) == 0.0
    assert float(summary_df.loc[0, "eval_overall_mean_f1"]) == 0.0

    leaderboard_df = pd.read_csv(result["leaderboard_path"])
    assert len(leaderboard_df) == 1
    assert leaderboard_df.loc[0, "model"] == "THRESHOLD_FLAT"

    experiment = mlflow.get_experiment_by_name("test-fusion-crops")
    assert experiment is not None
    child_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{result['mlflow_run_id']}'",
    )
    assert len(child_runs) == 1
    assert child_runs.loc[0, "params.model"] == "THRESHOLD_FLAT"
    assert child_runs.loc[0, "tags.run_kind"] == "model_run"
    assert float(child_runs.loc[0, "metrics.overall_mean_jaccard"]) == 0.0
    assert float(child_runs.loc[0, "metrics.overall_mean_f1"]) == 0.0


def test_run_crops_fusion_experiment_skips_weighted_models_without_weights(
    tmp_path: Path, monkeypatch
) -> None:
    seg = np.zeros((4, 4), dtype=np.uint8)
    seg[1:3, 1:3] = 1

    stack_a = tmp_path / "comp_a.tif"
    stack_b = tmp_path / "comp_b.tif"
    _write_stack(stack_a, seg)
    _write_stack(stack_b, seg)

    gt_full = np.zeros((10, 10), dtype=np.uint8)
    gt_full[3:5, 4:6] = 5
    gt_image = tmp_path / "gt.tif"
    tifffile.imwrite(gt_image, gt_full)

    qa_path = tmp_path / "qa.parquet"
    df = _make_qa_df(stack_a=stack_a, stack_b=stack_b, gt_image=gt_image, split="test")
    df.to_parquet(qa_path, index=False)

    output_dir = tmp_path / "out"
    model_dir = output_dir / "threshold_flat"
    model_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(model_dir / "fused_0000.tif", np.zeros((4, 4), dtype=np.uint8))

    fake_jar = tmp_path / "fake.jar"
    fake_jar.write_text("jar", encoding="utf-8")
    monkeypatch.setattr(ce, "DEFAULT_JAR_PATH", fake_jar)

    result = ce.run_crops_fusion_experiment(
        qa_parquet=qa_path,
        output_dir=output_dir,
        models=("THRESHOLD_FLAT", "THRESHOLD_USER"),
        skip_fusion=True,
        chunk_size=0,
        mlflow_experiment="test-fusion-crops-skip-weighted",
        mlflow_tracking_path=tmp_path / "mlruns",
    )

    assert result["models_run"] == ["THRESHOLD_FLAT"]
    assert result["weighted_models_skipped"] == ["THRESHOLD_USER"]

    summary_df = pd.read_csv(output_dir / "fusion_crops_summary.csv")
    assert sorted(summary_df["model"].tolist()) == ["THRESHOLD_FLAT"]


def test_run_crops_fusion_experiment_keeps_bic_flat_without_weights(
    tmp_path: Path, monkeypatch
) -> None:
    seg = np.zeros((4, 4), dtype=np.uint8)
    seg[1:3, 1:3] = 1

    stack_a = tmp_path / "comp_a.tif"
    stack_b = tmp_path / "comp_b.tif"
    _write_stack(stack_a, seg)
    _write_stack(stack_b, seg)

    gt_full = np.zeros((10, 10), dtype=np.uint8)
    gt_full[3:5, 4:6] = 5
    gt_image = tmp_path / "gt.tif"
    tifffile.imwrite(gt_image, gt_full)

    qa_path = tmp_path / "qa.parquet"
    df = _make_qa_df(stack_a=stack_a, stack_b=stack_b, gt_image=gt_image, split="test")
    df.to_parquet(qa_path, index=False)

    output_dir = tmp_path / "out"
    threshold_flat_dir = output_dir / "threshold_flat"
    bic_flat_dir = output_dir / "bic_flat_voting"
    threshold_flat_dir.mkdir(parents=True, exist_ok=True)
    bic_flat_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        threshold_flat_dir / "fused_0000.tif", np.zeros((4, 4), dtype=np.uint8)
    )
    tifffile.imwrite(bic_flat_dir / "fused_0000.tif", np.zeros((4, 4), dtype=np.uint8))

    fake_jar = tmp_path / "fake.jar"
    fake_jar.write_text("jar", encoding="utf-8")
    monkeypatch.setattr(ce, "DEFAULT_JAR_PATH", fake_jar)

    result = ce.run_crops_fusion_experiment(
        qa_parquet=qa_path,
        output_dir=output_dir,
        models=("THRESHOLD_FLAT", "BIC_FLAT_VOTING", "THRESHOLD_USER"),
        skip_fusion=True,
        chunk_size=0,
        mlflow_experiment="test-fusion-crops-keep-bic-flat",
        mlflow_tracking_path=tmp_path / "mlruns",
    )

    assert sorted(result["models_run"]) == ["BIC_FLAT_VOTING", "THRESHOLD_FLAT"]
    assert result["weighted_models_skipped"] == ["THRESHOLD_USER"]
