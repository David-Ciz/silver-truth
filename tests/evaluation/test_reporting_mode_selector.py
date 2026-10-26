import pandas as pd

from silver_truth.evaluation.reporting_mode_selector import (
    build_reporting_mode_candidates,
    validation_select_reporting_modes,
)


def _combined_row(
    *,
    dataset: str,
    fold: str,
    best_iou: float,
    selected_iou: float,
    validation_iou: float,
    method: str,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "crop_tag": "sz64",
        "variant": "baseline",
        "fold": fold,
        "row_type": "fold",
        "best_fusion_iou": 0.7,
        "best_competitor_iou": 0.8,
        "median_competitor_iou": 0.6,
        "worst_competitor_iou": 0.4,
        "silver_truth_iou": 0.85,
        "best_ours_iou": best_iou,
        "best_ours_method": method,
        "validation_selected_ours_iou": selected_iou,
        "validation_selected_ours_method": method,
        "validation_selected_validation_iou": validation_iou,
    }


def test_reporting_mode_selector_keeps_strict_taxonomy_and_selects_by_validation() -> None:
    full = pd.DataFrame(
        [
            _combined_row(
                dataset="Dataset",
                fold="fold-1",
                best_iou=0.95,
                selected_iou=0.70,
                validation_iou=0.75,
                method="full_pipeline_t0.70__simple",
            )
        ]
    )
    strict = pd.DataFrame(
        [
            _combined_row(
                dataset="Dataset",
                fold="fold-1",
                best_iou=0.80,
                selected_iou=0.79,
                validation_iou=0.82,
                method="ensemble_qa_t0.50",
            )
        ]
    )
    soft = pd.DataFrame(
        [
            {
                "dataset": "Dataset",
                "fold": "fold-1",
                "strategy": "pairwise",
                "preset": "balanced",
                "validation_mean_iou": 0.81,
                "test_mean_iou": 0.90,
                "test_mean_f1": 0.95,
                "input_path": "x",
                "output_dir": "y",
            }
        ]
    )

    candidates = build_reporting_mode_candidates(
        full_combined=full,
        strict_combined=strict,
        soft_strategy_summaries=[("pairwise_soft_hybrid", soft)],
    )
    selected = validation_select_reporting_modes(candidates)

    strict_row = candidates[candidates["mode"].eq("strict_qa_ensemble")].iloc[0]
    full_row = candidates[candidates["mode"].eq("full_deployable")].iloc[0]
    selected_fold = selected[selected["row_type"].eq("fold")].iloc[0]

    assert bool(strict_row["strict_table_eligible"])
    assert not bool(full_row["strict_table_eligible"])
    assert selected_fold["mode"] == "strict_qa_ensemble"
    assert selected_fold["selected_test_iou"] == 0.79
    assert selected[selected["row_type"].eq("mean")].iloc[0]["selected_test_iou"] == 0.79
