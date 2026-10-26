from pathlib import Path

import pandas as pd

from silver_truth.evaluation.ranker_pruned_fusion import (
    RankerPruningConfig,
    build_pruned_fusion_comparison_rows,
    build_ranker_pruned_candidate_table,
    build_ranker_pruned_fusion_context_table,
    infer_paper_input_metadata,
)
from scripts.run_ranker_pruned_non_simple_sweep import (
    select_best_backend_by_dataset,
    summarize_overall,
)
from scripts.build_ranker_pruned_validation_selection_tables import (
    select_by_dataset_validation,
    select_by_fold_validation,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_id": "a",
                "campaign_number": "01",
                "original_image_key": "t001",
                "label": 1,
                "competitor": "A",
                "stacked_path": "a.tif",
            },
            {
                "cell_id": "b",
                "campaign_number": "01",
                "original_image_key": "t001",
                "label": 1,
                "competitor": "B",
                "stacked_path": "b.tif",
            },
            {
                "cell_id": "c",
                "campaign_number": "01",
                "original_image_key": "t001",
                "label": 1,
                "competitor": "C",
                "stacked_path": "c.tif",
            },
            {
                "cell_id": "d",
                "campaign_number": "01",
                "original_image_key": "t002",
                "label": 2,
                "competitor": "A",
                "stacked_path": "d.tif",
            },
        ]
    )


def _ranked_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "ranker_cluster_id": "cluster-1",
                "candidate_source": "A",
                "pairwise_cluster_score": 0.70,
                "fusion_support": 4,
            },
            {
                "candidate_id": "b",
                "ranker_cluster_id": "cluster-1",
                "candidate_source": "B",
                "pairwise_cluster_score": 0.90,
                "fusion_support": 1,
            },
            {
                "candidate_id": "c",
                "ranker_cluster_id": "cluster-1",
                "candidate_source": "C",
                "pairwise_cluster_score": 0.86,
                "fusion_support": 3,
            },
            {
                "candidate_id": "d",
                "ranker_cluster_id": "cluster-2",
                "candidate_source": "A",
                "pairwise_cluster_score": 0.40,
                "fusion_support": 2,
            },
        ]
    )


def test_top2_per_cluster_writes_paper_ready_compatible_rows() -> None:
    filtered, stats = build_ranker_pruned_candidate_table(
        _candidate_rows(),
        _ranked_rows(),
        config=RankerPruningConfig(policy="top2_per_cluster"),
    )

    assert filtered["cell_id"].tolist() == ["b", "c", "d"]
    assert filtered["competitor"].tolist() == [
        "ranker_pruned_slot_01",
        "ranker_pruned_slot_02",
        "ranker_pruned_slot_01",
    ]
    assert filtered["original_competitor"].tolist() == ["B", "C", "A"]
    assert "stacked_path" in filtered.columns
    assert "pairwise_cluster_score" in filtered.columns
    assert stats.loc[0, "filtered_candidate_rows"] == 3
    assert stats.loc[0, "clusters_before"] == 2
    assert stats.loc[0, "clusters_after"] == 2


def test_margin_pruned_keeps_candidates_close_to_best_score() -> None:
    filtered, _ = build_ranker_pruned_candidate_table(
        _candidate_rows(),
        _ranked_rows(),
        config=RankerPruningConfig(policy="margin_pruned", margin=0.05),
    )

    assert filtered["cell_id"].tolist() == ["b", "c", "d"]


def test_fusion_safe_pruning_keeps_supported_candidate_unless_strongly_rejected() -> None:
    filtered, _ = build_ranker_pruned_candidate_table(
        _candidate_rows(),
        _ranked_rows(),
        config=RankerPruningConfig(
            policy="fusion_safe_pruning",
            margin=0.02,
            fusion_safe_margin=0.25,
        ),
    )

    assert filtered["cell_id"].tolist() == ["a", "b", "c", "d"]


def test_infer_paper_input_metadata_from_standard_path() -> None:
    metadata = infer_paper_input_metadata(
        Path(
            "hpc_results/msc/paper_runs/paper_inputs/"
            "Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-2_paper_ready.parquet"
        )
    )

    assert metadata["dataset"] == "Fluo-C2DL-MSC"
    assert metadata["crop_tag"] == "sz1024"
    assert metadata["variant"] == "msc_crop1024"
    assert metadata["fold"] == "fold-2"


def test_build_pruned_fusion_comparison_rows_uses_full_pipeline_taxonomy(
    tmp_path: Path,
) -> None:
    leaderboard_path = tmp_path / "leaderboard.csv"
    pd.DataFrame(
        [
            {
                "model": "SIMPLE",
                "eval_test_mean_jaccard": 0.8,
                "eval_test_mean_f1": 0.9,
                "selection_split": "test",
                "fusion_success": 1.0,
            }
        ]
    ).to_csv(leaderboard_path, index=False)
    pruning_summary = pd.DataFrame(
        [
            {
                "filtered_candidate_rows": 3,
                "input_candidate_rows": 4,
                "ranked_candidate_rows": 4,
                "matched_ranked_rows": 4,
                "clusters_before": 2,
                "clusters_after": 2,
            }
        ]
    )

    comparison = build_pruned_fusion_comparison_rows(
        leaderboard_path=leaderboard_path,
        candidate_parquet=Path(
            "data/paper_runs/paper_inputs/Dataset/sz64/baseline/"
            "fold-1_paper_ready.parquet"
        ),
        config=RankerPruningConfig(policy="top2_per_cluster"),
        pruning_summary=pruning_summary,
    )

    assert comparison.loc[0, "row_family"] == "full_pipeline"
    assert comparison.loc[0, "method_key"] == (
        "full_pipeline__qa_pruned_simple__top2_per_cluster"
    )
    assert comparison.loc[0, "mean_iou"] == 0.8


def test_pruned_fusion_method_key_records_score_and_topk(tmp_path: Path) -> None:
    leaderboard_path = tmp_path / "leaderboard.csv"
    pd.DataFrame(
        [
            {
                "model": "SIMPLE",
                "eval_test_mean_jaccard": 0.8,
                "eval_test_mean_f1": 0.9,
                "selection_split": "test",
                "fusion_success": 1.0,
            }
        ]
    ).to_csv(leaderboard_path, index=False)
    pruning_summary = pd.DataFrame([{}])

    comparison = build_pruned_fusion_comparison_rows(
        leaderboard_path=leaderboard_path,
        candidate_parquet=Path(
            "data/paper_runs/paper_inputs/Dataset/sz64/baseline/"
            "fold-1_paper_ready.parquet"
        ),
        config=RankerPruningConfig(
            policy="source_balanced_topk",
            score_column="source_train_mean_iou",
            top_k=5,
        ),
        pruning_summary=pruning_summary,
    )

    assert comparison.loc[0, "method_key"] == (
        "full_pipeline__qa_pruned_simple__source_balanced_topk__"
        "score_source_train_mean_iou__topk5"
    )
    assert comparison.loc[0, "score_column"] == "source_train_mean_iou"
    assert comparison.loc[0, "top_k"] == 5


def test_build_ranker_pruned_fusion_context_table_adds_baseline_rows() -> None:
    pruned = pd.DataFrame(
        [
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "row_family": "full_pipeline",
                "method_key": "full_pipeline__qa_pruned_simple__top2_per_cluster",
                "method_label": "QA-pruned SIMPLE",
                "policy": "top2_per_cluster",
                "fusion_backend": "SIMPLE",
                "mean_iou": 0.8,
                "mean_f1": 0.9,
            },
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-2",
                "row_family": "full_pipeline",
                "method_key": "full_pipeline__qa_pruned_simple__top2_per_cluster",
                "method_label": "QA-pruned SIMPLE",
                "policy": "top2_per_cluster",
                "fusion_backend": "SIMPLE",
                "mean_iou": 0.6,
                "mean_f1": 0.7,
            },
        ]
    )
    paper = pd.DataFrame(
        [
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "row_type": "fold",
                "best_fusion_iou": 0.7,
                "best_fusion_method": "fusion_only__simple",
                "best_competitor_iou": 0.85,
                "best_competitor_name": "CALT-US",
                "silver_truth_iou": 0.9,
            },
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-2",
                "row_type": "fold",
                "best_fusion_iou": 0.65,
                "best_fusion_method": "fusion_only__simple",
                "best_competitor_iou": 0.75,
                "best_competitor_name": "CALT-US",
                "silver_truth_iou": 0.8,
            },
        ]
    )
    soft = pd.DataFrame(
        [
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "selected_method": "pairwise:balanced",
                "selected_test_iou": 0.77,
            },
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-2",
                "selected_method": "pairwise:balanced",
                "selected_test_iou": 0.66,
            },
        ]
    )

    context = build_ranker_pruned_fusion_context_table(
        pruned_rows=pruned,
        paper_summary=paper,
        soft_hybrid_selection=soft,
    )

    fold_rows = context[context["row_type"].eq("fold")]
    assert set(fold_rows["row_family"]) == {
        "full_pipeline",
        "fusion_only",
        "competitor",
        "silver_truth",
        "reporting_mode_selected",
    }
    mean_pruned = context[
        context["row_type"].eq("mean")
        & context["method_key"].eq("full_pipeline__qa_pruned_simple__top2_per_cluster")
    ].iloc[0]
    assert mean_pruned["row_family"] == "full_pipeline"
    assert mean_pruned["mean_iou"] == 0.7
    assert mean_pruned["n_folds"] == 2


def test_build_ranker_pruned_fusion_context_table_deduplicates_shared_baselines() -> None:
    pruned = pd.DataFrame(
        [
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "method_key": "full_pipeline__qa_pruned_simple__top1_per_cluster",
                "method_label": "QA-pruned SIMPLE",
                "policy": "top1_per_cluster",
                "fusion_backend": "SIMPLE",
                "mean_iou": 0.6,
            },
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "method_key": "full_pipeline__qa_pruned_simple__top2_per_cluster",
                "method_label": "QA-pruned SIMPLE",
                "policy": "top2_per_cluster",
                "fusion_backend": "SIMPLE",
                "mean_iou": 0.7,
            },
        ]
    )
    paper = pd.DataFrame(
        [
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "row_type": "fold",
                "best_fusion_iou": 0.65,
                "best_fusion_method": "fusion_only__simple",
                "best_competitor_iou": 0.75,
                "best_competitor_name": "CALT-US",
                "silver_truth_iou": 0.8,
            }
        ]
    )

    context = build_ranker_pruned_fusion_context_table(
        pruned_rows=pruned,
        paper_summary=paper,
    )
    fold_fusion = context[
        context["row_type"].eq("fold")
        & context["row_family"].eq("fusion_only")
    ]

    assert len(fold_fusion) == 1


def test_non_simple_summary_selects_best_backend_by_dataset() -> None:
    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": "Dataset-A",
                "fusion_backend": "MAJORITY_FLAT",
                "mean_iou": 0.8,
                "mean_f1": 0.9,
            },
            {
                "dataset": "Dataset-A",
                "fusion_backend": "BIC_WEIGHTED_VOTING",
                "mean_iou": 0.82,
                "mean_f1": 0.91,
            },
            {
                "dataset": "Dataset-B",
                "fusion_backend": "MAJORITY_FLAT",
                "mean_iou": 0.7,
                "mean_f1": 0.8,
            },
            {
                "dataset": "Dataset-B",
                "fusion_backend": "BIC_WEIGHTED_VOTING",
                "mean_iou": 0.65,
                "mean_f1": 0.75,
            },
        ]
    )

    best = select_best_backend_by_dataset(dataset_summary)
    overall = summarize_overall(dataset_summary, best)

    assert best["fusion_backend"].tolist() == [
        "BIC_WEIGHTED_VOTING",
        "MAJORITY_FLAT",
    ]
    selected = overall[
        overall["selection_scope"].eq("best_backend_by_dataset")
    ].iloc[0]
    assert selected["fusion_backend"] == "dataset_selected"
    assert selected["mean_iou"] == 0.76


def test_ranker_pruned_validation_selection_uses_validation_not_test() -> None:
    rows = pd.DataFrame(
        [
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "policy": "policy_a",
                "validation_mean_iou": 0.90,
                "mean_iou": 0.60,
            },
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "policy": "policy_b",
                "validation_mean_iou": 0.80,
                "mean_iou": 0.95,
            },
        ]
    )

    selected = select_by_fold_validation(rows)

    assert selected["policy"].tolist() == ["policy_a"]
    assert selected["mean_iou"].tolist() == [0.60]


def test_ranker_pruned_dataset_selection_keeps_one_policy_per_dataset() -> None:
    rows = pd.DataFrame(
        [
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "policy": "policy_a",
                "validation_mean_iou": 0.90,
                "mean_iou": 0.70,
            },
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-2",
                "policy": "policy_a",
                "validation_mean_iou": 0.70,
                "mean_iou": 0.72,
            },
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-1",
                "policy": "policy_b",
                "validation_mean_iou": 0.81,
                "mean_iou": 0.80,
            },
            {
                "dataset": "Dataset",
                "crop_tag": "sz64",
                "variant": "baseline",
                "fold": "fold-2",
                "policy": "policy_b",
                "validation_mean_iou": 0.82,
                "mean_iou": 0.82,
            },
        ]
    )

    selected = select_by_dataset_validation(rows)

    assert selected["policy"].tolist() == ["policy_b", "policy_b"]
    assert selected["fold"].tolist() == ["fold-1", "fold-2"]
