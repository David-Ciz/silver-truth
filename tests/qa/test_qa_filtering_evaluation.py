import numpy as np
import pytest

from silver_truth.qa.filtering_evaluation import (
    calculate_filtering_metrics,
    parse_thresholds,
)


def test_parse_thresholds_sorts_and_deduplicates() -> None:
    thresholds = parse_thresholds("0.75,0.5,0.75,0.9")
    assert thresholds == [0.5, 0.75, 0.9]


def test_parse_thresholds_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        parse_thresholds("0.5,1.2")


def test_calculate_filtering_metrics_confusion_counts() -> None:
    y_true = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    y_pred = np.array([0.95, 0.4, 0.8, 0.2], dtype=float)

    metrics = calculate_filtering_metrics(y_true, y_pred, threshold=0.75)

    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["tn"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["fpr"] == 0.5
    assert metrics["fnr"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["filtered_count"] == 2.0
    assert metrics["filtered_pct"] == 50.0


def test_calculate_filtering_metrics_handles_nan_rows() -> None:
    y_true = np.array([0.9, np.nan, 0.3], dtype=float)
    y_pred = np.array([0.8, 0.1, np.nan], dtype=float)

    metrics = calculate_filtering_metrics(y_true, y_pred, threshold=0.75)
    assert metrics["n_samples"] == 1.0
    assert metrics["tp"] == 1.0
    assert metrics["filtered_count"] == 0.0
