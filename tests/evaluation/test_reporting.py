from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from silver_truth.evaluation.reporting import (
    analyze_overflow_impact,
    bootstrap_mean_ci,
    bootstrap_stratified_mean_ci,
    generate_hsc_reporting_bundle,
    generate_hsc_main_comparison_bundle,
    paired_wilcoxon_test,
)
from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    save_dataframe_to_parquet_with_metadata,
)


def test_bootstrap_mean_ci_handles_single_value() -> None:
    result = bootstrap_mean_ci([0.8], n_resamples=100, seed=1)
    assert result["mean"] == 0.8
    assert result["ci_lower"] == 0.8
    assert result["ci_upper"] == 0.8


def test_bootstrap_stratified_mean_ci_matches_fold_average() -> None:
    result = bootstrap_stratified_mean_ci(
        {
            "fold-1": [0.2, 0.4],
            "fold-2": [0.6, 0.8, 1.0],
        },
        n_resamples=100,
        seed=1,
    )
    assert result["mean"] == 0.55
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]


def test_paired_wilcoxon_test_handles_zero_differences() -> None:
    result = paired_wilcoxon_test([0.5, 0.7], [0.5, 0.7])
    assert result["n_pairs"] == 2
    assert result["p_value"] == 1.0
    assert result["note"] == "all paired differences are zero"


def test_generate_hsc_reporting_bundle_from_partial_fold_data(tmp_path: Path) -> None:
    paper_runs = tmp_path / "data" / "paper_runs"
    baselines = paper_runs / "baselines"
    baselines.mkdir(parents=True)

    competitor_df = pd.DataFrame(
        [
            {
                "competitor": "CALT-US",
                "campaign": 2,
                "image_key": "02_0001.tif",
                "label": 1,
                "jaccard_score": 0.80,
                "f1_score": 0.88,
                "image_average": 0.80,
                "image_f1_average": 0.88,
            },
            {
                "competitor": "DREX-US",
                "campaign": 2,
                "image_key": "02_0001.tif",
                "label": 1,
                "jaccard_score": 0.75,
                "f1_score": 0.85,
                "image_average": 0.75,
                "image_f1_average": 0.85,
            },
            {
                "competitor": "CALT-US",
                "campaign": 1,
                "image_key": "01_0002.tif",
                "label": 1,
                "jaccard_score": 0.10,
                "f1_score": 0.20,
                "image_average": 0.10,
                "image_f1_average": 0.20,
            },
        ]
    )
    competitor_df.to_csv(
        baselines / "BF-C2DL-HSC_sz64_baseline_fold-1_competitors.csv", index=False
    )

    silver_truth_df = competitor_df.iloc[[0]].copy()
    silver_truth_df["competitor"] = "SILVER-TRUTH"
    silver_truth_df["image_average"] = 0.82
    silver_truth_df["image_f1_average"] = 0.90
    silver_truth_df.to_csv(
        baselines / "BF-C2DL-HSC_sz64_baseline_fold-1_silver_truth.csv", index=False
    )

    ablation_root = (
        paper_runs / "ablation" / "BF-C2DL-HSC" / "sz64" / "baseline" / "fold-1"
    )
    _write_fullimage_eval(
        ablation_root / "qa_only" / "fullimage_eval.csv",
        iou=0.78,
        f1=0.87,
    )
    _write_fullimage_eval(
        ablation_root / "fusion_only" / "simple" / "fullimage_eval.csv",
        iou=0.81,
        f1=0.89,
    )
    _write_fullimage_eval(
        ablation_root / "full_pipeline_t0.75" / "simple" / "fullimage_eval.csv",
        iou=0.84,
        f1=0.91,
    )
    _write_ensemble_eval(
        ablation_root / "ensemble_only" / "C1_fake_set-test.parquet",
        iou=0.83,
        f1=0.90,
    )
    _write_ensemble_eval(
        ablation_root / "ensemble_qa_t0.75" / "C1_fake_set-test.parquet",
        iou=0.82,
        f1=0.89,
    )

    bundle = generate_hsc_reporting_bundle(
        paper_runs_root=paper_runs,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    inventory = bundle["inventory"]
    assert inventory["exists"].sum() == 7

    core_summary = bundle["core_summary"]
    assert "CALT-US (best competitor)" in set(core_summary["method_label"])
    assert "fusion_only / SIMPLE" in set(core_summary["method_label"])
    assert "full_pipeline / SIMPLE @ t=0.75" in set(core_summary["method_label"])

    comparisons = bundle["comparisons"]
    assert not comparisons.empty
    assert "candidate_method_label" in comparisons.columns
    assert "reference_method_label" in comparisons.columns
    assert (
        (
            comparisons["candidate_method_label"] == "ensemble_only"
        )
        & (comparisons["paired_images"] == 1)
    ).any()


def test_analyze_overflow_impact_splits_cells_by_crop_overflow(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    gt_dir = data_root / "synchronized_data" / "BF-C2DL-MuSC" / "01_GT" / "SEG"
    gt_dir.mkdir(parents=True)

    gt_fit = np.zeros((12, 12), dtype=np.uint16)
    gt_fit[2:6, 2:6] = 1
    gt_overflow = np.zeros((12, 12), dtype=np.uint16)
    gt_overflow[1:10, 5:7] = 1

    gt_fit_path = gt_dir / "man_seg0001.tif"
    gt_overflow_path = gt_dir / "man_seg0002.tif"
    tifffile.imwrite(gt_fit_path, gt_fit)
    tifffile.imwrite(gt_overflow_path, gt_overflow)

    dataset_parquet = (
        data_root / "dataframes" / "BF-C2DL-MuSC" / "whole_image" / "sample.parquet"
    )
    dataset_df = pd.DataFrame(
        [
            {
                "gt_image": "data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/man_seg0001.tif",
                "campaign_number": "01",
                "composite_key": "01_0001.tif",
                "split": "test",
            },
            {
                "gt_image": "data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/man_seg0002.tif",
                "campaign_number": "01",
                "composite_key": "01_0002.tif",
                "split": "test",
            },
        ]
    )
    save_dataframe_to_parquet_with_metadata(dataset_df, str(dataset_parquet))

    results_csv = tmp_path / "detailed_results.csv"
    pd.DataFrame(
        [
            {
                "competitor": "CALT-US",
                "gt_seg_path": "data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/man_seg0001.tif",
                "cell_id": 1,
                "jaccard_score": 0.9,
                "f1_score": 0.95,
            },
            {
                "competitor": "CALT-US",
                "gt_seg_path": "data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/man_seg0002.tif",
                "cell_id": 1,
                "jaccard_score": 0.2,
                "f1_score": 0.33,
            },
        ]
    ).to_csv(results_csv, index=False)

    bundle = analyze_overflow_impact(
        results_path=results_csv,
        dataset_dataframe_path=dataset_parquet,
        crop_size=6,
    )

    enriched = bundle["enriched"]
    assert set(enriched["overflow_status"]) == {"fits", "overflow"}

    overall = bundle["overall_summary"].sort_values("overflow_status").reset_index(drop=True)
    assert overall.loc[0, "overflow_status"] == "fits"
    assert overall.loc[0, "jaccard_score_mean"] == 0.9
    assert overall.loc[1, "overflow_status"] == "overflow"
    assert overall.loc[1, "jaccard_score_mean"] == 0.2

    by_comp = bundle["by_competitor_summary"]
    assert set(by_comp["competitor"]) == {"CALT-US"}


def test_generate_hsc_main_comparison_bundle(tmp_path: Path) -> None:
    comparison_root = tmp_path / "clean_comparison"
    for fold, campaign_number, image_prefix, iou in (
        ("fold-1", 2, "t0001", 0.83),
        ("fold-2", 1, "t0002", 0.81),
    ):
        fold_dir = comparison_root / fold
        fold_dir.mkdir(parents=True)

        competitor_df = pd.DataFrame(
            [
                {
                    "competitor": "CALT-US",
                    "campaign": campaign_number,
                    "image_key": f"{campaign_number:02d}_{image_prefix}.tif",
                    "label": 1,
                    "jaccard_score": iou - 0.05,
                    "f1_score": 0.90,
                    "image_average": iou - 0.05,
                    "image_f1_average": 0.90,
                }
            ]
        )
        competitor_df.to_csv(fold_dir / "competitors.csv", index=False)

        silver_truth_df = competitor_df.copy()
        silver_truth_df["competitor"] = "SILVER-TRUTH"
        silver_truth_df["image_average"] = iou - 0.02
        silver_truth_df["image_f1_average"] = 0.91
        silver_truth_df.to_csv(fold_dir / "silver_truth.csv", index=False)

        _write_ensemble_eval(
            fold_dir / "checkpoints" / "C1_ds1-42-7015_QA--_M2--_set-test.parquet",
            campaign_number=campaign_number,
            image_key=image_prefix,
            iou=iou,
            f1=0.92,
        )

    bundle = generate_hsc_main_comparison_bundle(
        comparison_root=comparison_root,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    core_summary = bundle["core_summary"]
    assert set(core_summary["method_label"]) == {
        "CALT-US (best competitor)",
        "SILVER-TRUTH",
        "ensemble_only",
    }

    comparisons = bundle["comparisons"]
    assert set(comparisons["metric"]) == {"iou", "f1"}
    assert (comparisons["candidate_method_key"] == "ensemble_only").all()

    limitations = bundle["limitations"]
    assert "descriptive_only" in set(limitations["status"])


def _write_fullimage_eval(path: Path, *, iou: float, f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "campaign_number": 2,
                "original_image_key": "t0001",
                "gt_image": "gt.tif",
                "reconstructed_path": "pred.tif",
                "split": "test",
                "cells_placed": 1,
                "iou": iou,
                "f1": f1,
            }
        ]
    )
    df.to_csv(path, index=False)


def _write_ensemble_eval(
    path: Path,
    *,
    iou: float,
    f1: float,
    campaign_number: int = 2,
    image_key: str = "t0001",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "campaign_number": f"c{campaign_number:02d}",
                "original_image_key": image_key,
                "gt_image": "gt.tif",
                "reconstructed_path": "pred.tif",
                "split": "test",
                "cells_placed": 1,
                "iou": iou,
                "f1": f1,
            }
        ]
    )
    df.to_parquet(path, index=False)
