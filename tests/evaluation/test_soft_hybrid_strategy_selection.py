from pathlib import Path

import pandas as pd

from silver_truth.evaluation.soft_hybrid_strategy_selection import (
    load_strategy_summaries,
    summarize_selected_by_dataset,
    validation_select_strategies,
)


def test_validation_select_strategies_uses_validation_not_test(tmp_path: Path) -> None:
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    input_path = "root/paper_inputs/Dataset/sz64/variant/fold-1_paper_ready.parquet"
    pd.DataFrame(
        [
            {
                "input_path": input_path,
                "output_dir": "a",
                "preset": "balanced",
                "split": "validation",
                "mean_iou": 0.8,
                "mean_f1": 0.9,
            },
            {
                "input_path": input_path,
                "output_dir": "a",
                "preset": "balanced",
                "split": "test",
                "mean_iou": 0.9,
                "mean_f1": 0.95,
            },
        ]
    ).to_csv(path_a, index=False)
    pd.DataFrame(
        [
            {
                "input_path": input_path,
                "output_dir": "b",
                "preset": "balanced",
                "split": "validation",
                "mean_iou": 0.85,
                "mean_f1": 0.91,
            },
            {
                "input_path": input_path,
                "output_dir": "b",
                "preset": "balanced",
                "split": "test",
                "mean_iou": 0.7,
                "mean_f1": 0.8,
            },
        ]
    ).to_csv(path_b, index=False)

    summary = load_strategy_summaries([("a", path_a), ("b", path_b)])
    selected = validation_select_strategies(summary)
    dataset_summary = summarize_selected_by_dataset(selected)

    assert selected.iloc[0]["strategy"] == "b"
    assert selected.iloc[0]["test_mean_iou"] == 0.7
    assert dataset_summary.iloc[0]["mean_test_iou"] == 0.7


def test_validation_select_strategies_prefers_conservative_preset_within_epsilon() -> None:
    input_path = "root/paper_inputs/Dataset/sz64/variant/fold-1_paper_ready.parquet"
    summary = pd.DataFrame(
        [
            {
                "dataset": "Dataset",
                "fold": "fold-1",
                "strategy": "pairwise",
                "input_path": input_path,
                "output_dir": "balanced",
                "preset": "balanced",
                "split": "validation",
                "mean_iou": 0.840,
                "mean_f1": 0.9,
            },
            {
                "dataset": "Dataset",
                "fold": "fold-1",
                "strategy": "pairwise",
                "input_path": input_path,
                "output_dir": "balanced",
                "preset": "balanced",
                "split": "test",
                "mean_iou": 0.670,
                "mean_f1": 0.8,
            },
            {
                "dataset": "Dataset",
                "fold": "fold-1",
                "strategy": "pairwise",
                "input_path": input_path,
                "output_dir": "temporal",
                "preset": "temporal_heavy",
                "split": "validation",
                "mean_iou": 0.844,
                "mean_f1": 0.91,
            },
            {
                "dataset": "Dataset",
                "fold": "fold-1",
                "strategy": "pairwise",
                "input_path": input_path,
                "output_dir": "temporal",
                "preset": "temporal_heavy",
                "split": "test",
                "mean_iou": 0.660,
                "mean_f1": 0.79,
            },
        ]
    )

    selected = validation_select_strategies(summary, validation_epsilon=0.005)

    assert selected.iloc[0]["preset"] == "balanced"
    assert selected.iloc[0]["validation_mean_iou"] == 0.840
