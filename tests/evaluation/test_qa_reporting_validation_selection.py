import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from silver_truth.cli.evaluation import (
    combine_paper_result_summaries,
    score_parquet,
    select_ablation_threshold,
    summarize_paper_results,
)
from silver_truth.evaluation.qa_reporting import (
    _summarize_ablation_outputs,
    _validation_selected_full_pipeline,
)


def test_validation_selected_full_pipeline_uses_validation_iou_not_test_iou() -> None:
    summary = pd.DataFrame(
        [
            {
                "method_key": "full_pipeline_t0.40__simple",
                "family": "full_pipeline",
                "threshold": 0.40,
                "validation_mean_iou": 0.80,
                "validation_mean_f1": 0.88,
                "test_mean_iou": 0.90,
            },
            {
                "method_key": "full_pipeline_t0.75__simple",
                "family": "full_pipeline",
                "threshold": 0.75,
                "validation_mean_iou": 0.82,
                "validation_mean_f1": 0.87,
                "test_mean_iou": 0.70,
            },
            {
                "method_key": "fusion_only__simple",
                "family": "fusion_only",
                "threshold": None,
                "validation_mean_iou": 0.95,
                "validation_mean_f1": 0.96,
                "test_mean_iou": 0.95,
            },
        ]
    )

    selected = _validation_selected_full_pipeline(summary)

    assert selected is not None
    assert selected["method_key"] == "full_pipeline_t0.75__simple"
    assert selected["test_mean_iou"] == 0.70


def test_validation_selected_full_pipeline_can_be_restricted_to_fuser() -> None:
    summary = pd.DataFrame(
        [
            {
                "method_key": "full_pipeline_t0.60__bic_flat_voting",
                "family": "full_pipeline",
                "fusion_model": "bic_flat_voting",
                "threshold": 0.60,
                "validation_mean_iou": 0.90,
                "validation_mean_f1": 0.90,
                "test_mean_iou": 0.10,
            },
            {
                "method_key": "full_pipeline_t0.75__simple",
                "family": "full_pipeline",
                "fusion_model": "simple",
                "threshold": 0.75,
                "validation_mean_iou": 0.80,
                "validation_mean_f1": 0.89,
                "test_mean_iou": 0.70,
            },
        ]
    )

    selected = _validation_selected_full_pipeline(summary, fusion_model="simple")

    assert selected is not None
    assert selected["method_key"] == "full_pipeline_t0.75__simple"


def _write_fullimage_eval(path: Path, validation_iou: float, test_iou: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "campaign_number": 1,
                "original_image_key": "t0001",
                "split": "validation",
                "iou": validation_iou,
                "f1": 0.90,
                "cells_placed": 3,
            },
            {
                "campaign_number": 2,
                "original_image_key": "t0002",
                "split": "test",
                "iou": test_iou,
                "f1": 0.91,
                "cells_placed": 4,
            },
        ]
    ).to_csv(path, index=False)


def test_select_ablation_threshold_cli_writes_selected_artifacts(
    tmp_path: Path,
) -> None:
    ablation_dir = tmp_path / "fold-1"
    _write_fullimage_eval(
        ablation_dir / "full_pipeline_t0.40" / "simple" / "fullimage_eval.csv",
        validation_iou=0.80,
        test_iou=0.95,
    )
    _write_fullimage_eval(
        ablation_dir / "full_pipeline_t0.70" / "simple" / "fullimage_eval.csv",
        validation_iou=0.82,
        test_iou=0.70,
    )

    output_json = tmp_path / "selected.json"
    output_threshold = tmp_path / "selected.txt"
    output_label = tmp_path / "selected_label.txt"

    result = CliRunner().invoke(
        select_ablation_threshold,
        [
            "--ablation-dir",
            str(ablation_dir),
            "--fusion-model",
            "simple",
            "--output-json",
            str(output_json),
            "--output-threshold",
            str(output_threshold),
            "--output-label",
            str(output_label),
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_json.read_text())
    assert record["threshold"] == 0.70
    assert record["method_key"] == "full_pipeline_t0.70__simple"
    assert output_threshold.read_text().strip() == "0.700000"
    assert output_label.read_text().strip() == "0.70"


def test_ensemble_validation_and_test_parquets_are_merged(
    tmp_path: Path,
) -> None:
    ablation_dir = tmp_path / "fold-1"
    ensemble_dir = ablation_dir / "ensemble_only"
    ensemble_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"split": "validation", "iou": 0.80, "f1": 0.88, "cells_placed": 3},
            {"split": "validation", "iou": 0.90, "f1": 0.94, "cells_placed": 4},
        ]
    ).to_parquet(ensemble_dir / "C1_fake_set-validation.parquet", index=False)
    pd.DataFrame(
        [
            {"split": "test", "iou": 0.70, "f1": 0.82, "cells_placed": 5},
            {"split": "test", "iou": 0.74, "f1": 0.85, "cells_placed": 6},
        ]
    ).to_parquet(ensemble_dir / "C1_fake_set-test.parquet", index=False)

    summary = _summarize_ablation_outputs(ablation_dir)

    row = summary[summary["method_key"] == "ensemble_only"].iloc[0]
    assert len(summary[summary["method_key"] == "ensemble_only"]) == 1
    assert row["validation_n_images"] == 2
    assert row["test_n_images"] == 2
    assert row["validation_mean_iou"] == pytest.approx(0.85)
    assert row["test_mean_iou"] == pytest.approx(0.72)
    assert row["mean_iou"] == pytest.approx(0.72)


def test_fullimage_summary_uses_label_weighted_metrics_when_available(
    tmp_path: Path,
) -> None:
    ablation_dir = tmp_path / "fold-1"
    output_path = ablation_dir / "fusion_only" / "simple" / "fullimage_eval.csv"
    output_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "split": "test",
                "iou": 0.20,
                "f1": 0.30,
                "labels_scored": 1,
                "cells_placed": 1,
            },
            {
                "split": "test",
                "iou": 1.00,
                "f1": 1.00,
                "labels_scored": 9,
                "cells_placed": 9,
            },
        ]
    ).to_csv(output_path, index=False)

    summary = _summarize_ablation_outputs(ablation_dir)

    row = summary[summary["method_key"] == "fusion_only__simple"].iloc[0]
    assert row["test_n_images"] == 2
    assert row["test_mean_iou"] == pytest.approx(0.92)
    assert row["test_mean_f1"] == pytest.approx(0.93)
    assert row["mean_iou"] == pytest.approx(0.92)


def test_summarize_ablation_outputs_includes_soft_hybrid_fullimage_csv(
    tmp_path: Path,
) -> None:
    ablation_dir = tmp_path / "fold-1"
    output_path = ablation_dir / "soft_hybrid" / "fullimage_eval.csv"
    _write_fullimage_eval(output_path, validation_iou=0.82, test_iou=0.86)

    summary = _summarize_ablation_outputs(ablation_dir)

    row = summary[summary["method_key"] == "soft_hybrid"].iloc[0]
    assert row["family"] == "soft_hybrid"
    assert bool(row["uses_qa"])
    assert pd.isna(row["threshold"])
    assert pd.isna(row["fusion_model"])
    assert row["validation_mean_iou"] == pytest.approx(0.82)
    assert row["test_mean_iou"] == pytest.approx(0.86)


def test_soft_hybrid_method_name_is_parsed_after_fold_directory(
    tmp_path: Path,
) -> None:
    ablation_dir = tmp_path / "soft_hybrid_grid_smoke"
    output_root = ablation_dir / "Dataset" / "sz64" / "baseline" / "fold-1"
    _write_fullimage_eval(
        output_root / "soft_hybrid_balanced" / "fullimage_eval.csv",
        validation_iou=0.82,
        test_iou=0.86,
    )
    _write_fullimage_eval(
        output_root / "soft_hybrid_fusion_heavy" / "fullimage_eval.csv",
        validation_iou=0.84,
        test_iou=0.88,
    )

    summary = _summarize_ablation_outputs(ablation_dir)

    rows = summary[summary["family"] == "soft_hybrid"].sort_values("method_key")
    assert rows["method_key"].tolist() == [
        "soft_hybrid_balanced",
        "soft_hybrid_fusion_heavy",
    ]
    assert rows["test_mean_iou"].tolist() == pytest.approx([0.86, 0.88])


def test_score_parquet_oracle_writes_true_quality_score(tmp_path: Path) -> None:
    source = tmp_path / "paper_ready.parquet"
    output = tmp_path / "oracle.parquet"
    pd.DataFrame(
        {
            "jaccard_score": [0.25, 0.75],
            "predicted_jaccard_index": [0.1, 0.2],
        }
    ).to_parquet(source, index=False)

    result = CliRunner().invoke(
        score_parquet,
        [
            str(source),
            "--mode",
            "oracle",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    scored = pd.read_parquet(output)
    assert scored["predicted_jaccard_index"].tolist() == [0.25, 0.75]


def test_score_parquet_competitor_prior_uses_train_split_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paper_ready.parquet"
    output = tmp_path / "prior.parquet"
    pd.DataFrame(
        {
            "split": ["train", "train", "validation", "test"],
            "competitor": ["A", "B", "A", "B"],
            "jaccard_score": [0.8, 0.2, 0.1, 0.9],
        }
    ).to_parquet(source, index=False)

    result = CliRunner().invoke(
        score_parquet,
        [
            str(source),
            "--mode",
            "competitor_prior",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    scored = pd.read_parquet(output)
    assert scored["predicted_jaccard_index"].tolist() == [0.8, 0.2, 0.8, 0.2]


def test_summarize_paper_results_writes_final_table_csv(tmp_path: Path) -> None:
    paper_runs = tmp_path / "paper_runs"
    report_dir = (
        paper_runs
        / "reports"
        / "BF-C2DL-TEST"
        / "sz64"
        / "baseline"
        / "fold-1"
        / "ablation_diagnostics"
    )
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method_key": "fusion_only__simple",
                "family": "fusion_only",
                "test_mean_iou": 0.70,
                "test_mean_f1": 0.82,
                "validation_mean_iou": 0.71,
                "validation_mean_f1": 0.83,
            },
            {
                "method_key": "full_pipeline_t0.40__simple",
                "family": "full_pipeline",
                "test_mean_iou": 0.80,
                "test_mean_f1": 0.88,
                "validation_mean_iou": 0.72,
                "validation_mean_f1": 0.84,
            },
            {
                "method_key": "ensemble_only",
                "family": "ensemble_only",
                "test_mean_iou": 0.75,
                "test_mean_f1": 0.86,
                "validation_mean_iou": 0.90,
                "validation_mean_f1": 0.95,
            },
            {
                "method_key": "oracle_qa_only",
                "family": "oracle_qa_only",
                "test_mean_iou": 0.95,
                "test_mean_f1": 0.97,
                "validation_mean_iou": 0.96,
                "validation_mean_f1": 0.98,
            },
            {
                "method_key": "competitor_prior_qa_only",
                "family": "competitor_prior_qa_only",
                "test_mean_iou": 0.78,
                "test_mean_f1": 0.87,
                "validation_mean_iou": 0.77,
                "validation_mean_f1": 0.86,
            },
        ]
    ).to_csv(report_dir / "ablation_method_summary.csv", index=False)

    baselines = paper_runs / "baselines"
    baselines.mkdir()
    pd.DataFrame(
        {
            "competitor": ["A", "B", "C"],
            "split_test_average": [0.60, 0.85, 0.40],
        }
    ).to_csv(
        baselines / "BF-C2DL-TEST_sz64_baseline_fold-1_competitors.csv",
        index=False,
    )
    pd.DataFrame(
        {"competitor": ["SILVER-TRUTH"], "split_test_average": [0.88]}
    ).to_csv(
        baselines / "BF-C2DL-TEST_sz64_baseline_fold-1_silver_truth.csv",
        index=False,
    )
    output = tmp_path / "summary.csv"

    result = CliRunner().invoke(
        summarize_paper_results,
        [
            "--paper-runs-root",
            str(paper_runs),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = pd.read_csv(output)
    fold = summary[summary["row_type"] == "fold"].iloc[0]
    assert fold["best_competitor_iou"] == pytest.approx(0.85)
    assert fold["median_competitor_iou"] == pytest.approx(0.60)
    assert fold["worst_competitor_iou"] == pytest.approx(0.40)
    assert fold["silver_truth_iou"] == pytest.approx(0.88)
    assert fold["best_ours_iou"] == pytest.approx(0.80)
    assert fold["best_ours_method"] == "full_pipeline_t0.40__simple"
    assert fold["validation_selected_ours_iou"] == pytest.approx(0.75)
    assert fold["validation_selected_ours_method"] == "ensemble_only"
    assert fold["oracle_bound_iou"] == pytest.approx(0.95)
    assert fold["competitor_prior_iou"] == pytest.approx(0.78)


def test_combine_paper_result_summaries_writes_all_dataset_table(
    tmp_path: Path,
) -> None:
    columns = [
        "dataset",
        "crop_tag",
        "variant",
        "fold",
        "row_type",
        "best_fusion_iou",
        "best_fusion_method",
        "best_competitor_iou",
        "best_competitor_name",
        "median_competitor_iou",
        "median_competitor_name",
        "worst_competitor_iou",
        "worst_competitor_name",
        "silver_truth_iou",
        "best_ours_iou",
        "best_ours_method",
        "validation_selected_ours_iou",
        "validation_selected_ours_method",
        "validation_selected_validation_iou",
        "oracle_bound_iou",
        "competitor_prior_iou",
        "ablation_method_summary_path",
    ]
    first = pd.DataFrame(
        [
            {
                "dataset": "BF-C2DL-HSC",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "mean",
                "row_type": "mean",
                "best_fusion_iou": 0.87,
                "best_fusion_method": "foldwise best",
                "best_competitor_iou": 0.88,
                "best_competitor_name": "CALT-US",
                "median_competitor_iou": 0.80,
                "median_competitor_name": "",
                "worst_competitor_iou": 0.65,
                "worst_competitor_name": "DREX-US",
                "silver_truth_iou": 0.89,
                "best_ours_iou": 0.86,
                "best_ours_method": "foldwise best",
                "validation_selected_ours_iou": 0.84,
                "validation_selected_ours_method": "validation-selected per fold",
                "validation_selected_validation_iou": 0.90,
                "oracle_bound_iou": 0.91,
                "competitor_prior_iou": 0.88,
                "ablation_method_summary_path": "",
            }
        ],
        columns=columns,
    )
    second = first.copy()
    second.loc[0, "dataset"] = "DIC-C2DH-HeLa"
    second.loc[0, "best_ours_iou"] = 0.93
    duplicate_override = second.copy()
    duplicate_override.loc[0, "best_ours_iou"] = 0.94

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    override_path = tmp_path / "override.csv"
    output = tmp_path / "combined.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)
    duplicate_override.to_csv(override_path, index=False)

    result = CliRunner().invoke(
        combine_paper_result_summaries,
        [
            str(first_path),
            str(second_path),
            str(override_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    combined = pd.read_csv(output)
    assert combined["dataset"].tolist() == ["BF-C2DL-HSC", "DIC-C2DH-HeLa"]
    hela = combined[combined["dataset"] == "DIC-C2DH-HeLa"].iloc[0]
    assert hela["best_ours_iou"] == pytest.approx(0.94)
