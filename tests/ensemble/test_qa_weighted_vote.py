from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from scripts.run_qa_weighted_object_vote import _discovered_jobs
from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    SILVER_TRUTH_COLUMN,
)
from silver_truth.ensemble.qa_weighted_vote import run_qa_weighted_object_vote


tifffile = pytest.importorskip("tifffile")


def _write_stack(path: Path, mask: np.ndarray, gt: np.ndarray) -> None:
    raw = np.full(mask.shape, 100, dtype=np.uint8)
    stack = np.stack([raw, mask.astype(np.uint8), gt.astype(np.uint8)], axis=0)
    tifffile.imwrite(path, stack)


def test_qa_weighted_object_vote_prefers_high_weight_candidate(tmp_path: Path) -> None:
    good_mask = np.zeros((5, 5), dtype=np.uint8)
    good_mask[1:4, 1:4] = 1
    bad_mask = np.zeros((5, 5), dtype=np.uint8)
    bad_mask[0:2, 0:2] = 1
    gt = good_mask.copy()

    good_stack = tmp_path / "good.tif"
    bad_stack = tmp_path / "bad.tif"
    _write_stack(good_stack, good_mask, gt)
    _write_stack(bad_stack, bad_mask, gt)

    parquet = tmp_path / "fold-1_paper_ready_top2_per_cluster.parquet"
    pd.DataFrame(
        [
            {
                "cell_id": "good",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "ranker_pruned_slot_01",
                "stacked_path": str(good_stack),
                "split": "test",
                "pairwise_cluster_score": 0.95,
            },
            {
                "cell_id": "bad",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "ranker_pruned_slot_02",
                "stacked_path": str(bad_stack),
                "split": "test",
                "pairwise_cluster_score": 0.05,
            },
        ]
    ).to_parquet(parquet, index=False)

    result = run_qa_weighted_object_vote(
        candidate_parquet=parquet,
        output_dir=tmp_path / "out",
        weights_column="pairwise_cluster_score",
        threshold=0.50,
    )

    eval_df = pd.read_csv(result["eval_path"])
    comparison = pd.read_csv(result["comparison_path"])
    fused = tifffile.imread(tmp_path / "out" / "fused" / "fused_0000.tif")

    assert np.array_equal(fused > 0, good_mask > 0)
    assert float(eval_df.loc[0, "jaccard"]) == 1.0
    assert comparison.loc[0, "row_family"] == "ensemble_qa"
    assert comparison.loc[0, "method_key"] == "ensemble_qa_weighted_object_vote"
    assert float(comparison.loc[0, "mean_iou"]) == 1.0


def test_qa_weighted_object_vote_can_keep_top_k_per_group(tmp_path: Path) -> None:
    good_mask = np.zeros((5, 5), dtype=np.uint8)
    good_mask[1:4, 1:4] = 1
    bad_mask = np.ones((5, 5), dtype=np.uint8)
    gt = good_mask.copy()

    good_stack = tmp_path / "good.tif"
    bad_stack = tmp_path / "bad.tif"
    _write_stack(good_stack, good_mask, gt)
    _write_stack(bad_stack, bad_mask, gt)

    parquet = tmp_path / "fold-1_paper_ready_full_pool.parquet"
    pd.DataFrame(
        [
            {
                "cell_id": "bad",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "bad",
                "stacked_path": str(bad_stack),
                "split": "test",
                "ranker_cluster_id": "cluster-1",
                "candidate_set_weight_score": 0.1,
            },
            {
                "cell_id": "good",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "good",
                "stacked_path": str(good_stack),
                "split": "test",
                "ranker_cluster_id": "cluster-1",
                "candidate_set_weight_score": 0.9,
            },
        ]
    ).to_parquet(parquet, index=False)

    result = run_qa_weighted_object_vote(
        candidate_parquet=parquet,
        output_dir=tmp_path / "out",
        weights_column="candidate_set_weight_score",
        top_k_per_group=1,
        threshold=0.50,
    )

    eval_df = pd.read_csv(result["eval_path"])
    comparison = pd.read_csv(result["comparison_path"])
    fused = tifffile.imread(tmp_path / "out" / "fused" / "fused_0000.tif")

    assert np.array_equal(fused > 0, good_mask > 0)
    assert int(eval_df.loc[0, "candidates_used"]) == 1
    assert int(comparison.loc[0, "top_k_per_group"]) == 1
    assert comparison.loc[0, "method_key"] == "ensemble_qa_weighted_topk_object_vote"
    assert float(comparison.loc[0, "mean_iou"]) == 1.0


def test_qa_weighted_object_vote_ignores_silver_truth_rows(tmp_path: Path) -> None:
    candidate_mask = np.ones((3, 3), dtype=np.uint8)
    silver_mask = np.zeros((3, 3), dtype=np.uint8)
    gt = candidate_mask.copy()

    candidate_stack = tmp_path / "candidate.tif"
    silver_stack = tmp_path / "silver.tif"
    _write_stack(candidate_stack, candidate_mask, gt)
    _write_stack(silver_stack, silver_mask, gt)

    parquet = tmp_path / "fold-1_paper_ready_top1.parquet"
    pd.DataFrame(
        [
            {
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "ranker_pruned_slot_01",
                "stacked_path": str(candidate_stack),
                "split": "test",
                "pairwise_cluster_score": 0.5,
            },
                {
                    "campaign_number": "01",
                    "original_image_key": "t0001",
                    "label": 1,
                    "competitor": SILVER_TRUTH_COLUMN,
                    "stacked_path": str(silver_stack),
                    "split": "test",
                    "pairwise_cluster_score": 100.0,
                },
        ]
    ).to_parquet(parquet, index=False)

    result = run_qa_weighted_object_vote(
        candidate_parquet=parquet,
        output_dir=tmp_path / "out",
        weights_column="pairwise_cluster_score",
    )

    eval_df = pd.read_csv(result["eval_path"])
    assert int(eval_df.loc[0, "candidates_used"]) == 1
    assert float(eval_df.loc[0, "jaccard"]) == 1.0


def test_qa_weighted_object_vote_accepts_explicit_uniform_weights(
    tmp_path: Path,
) -> None:
    mask_a = np.zeros((3, 3), dtype=np.uint8)
    mask_a[0, 0] = 1
    mask_b = np.zeros((3, 3), dtype=np.uint8)
    mask_b[2, 2] = 1
    gt = np.logical_or(mask_a, mask_b).astype(np.uint8)

    stack_a = tmp_path / "a.tif"
    stack_b = tmp_path / "b.tif"
    _write_stack(stack_a, mask_a, gt)
    _write_stack(stack_b, mask_b, gt)

    parquet = tmp_path / "fold-1_paper_ready_top2.parquet"
    pd.DataFrame(
        [
            {
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "a",
                "stacked_path": str(stack_a),
                "split": "test",
            },
            {
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "b",
                "stacked_path": str(stack_b),
                "split": "test",
            },
        ]
    ).to_parquet(parquet, index=False)

    result = run_qa_weighted_object_vote(
        candidate_parquet=parquet,
        output_dir=tmp_path / "out",
        weights_column="uniform",
        threshold=0.50,
    )

    comparison = pd.read_csv(result["comparison_path"])
    assert comparison.loc[0, "weights_column"] == "_uniform_weight"
    assert comparison.loc[0, "row_family"] == "ensemble_only"
    assert comparison.loc[0, "method_key"] == "ensemble_only_uniform_object_vote"
    assert float(comparison.loc[0, "mean_iou"]) == 1.0


def test_boundary_aware_object_vote_downweights_weak_boundary(
    tmp_path: Path,
) -> None:
    gt = np.zeros((5, 5), dtype=np.uint8)
    gt[1:4, 1:4] = 1
    oversegmented = np.ones((5, 5), dtype=np.uint8)

    good_stack = tmp_path / "good.tif"
    over_stack = tmp_path / "over.tif"
    _write_stack(good_stack, gt, gt)
    _write_stack(over_stack, oversegmented, gt)

    parquet = tmp_path / "fold-1_paper_ready.parquet"
    pd.DataFrame(
        [
            {
                "cell_id": "good",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "good",
                "stacked_path": str(good_stack),
                "split": "test",
            },
            {
                "cell_id": "over",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "over",
                "stacked_path": str(over_stack),
                "split": "test",
            },
        ]
    ).to_parquet(parquet, index=False)
    feature_parquet = tmp_path / "features.parquet"
    pd.DataFrame(
        [
            {"candidate_id": "good", "local_image_support_score": 1.0},
            {"candidate_id": "over", "local_image_support_score": 0.0},
        ]
    ).to_parquet(feature_parquet, index=False)

    result = run_qa_weighted_object_vote(
        candidate_parquet=parquet,
        feature_parquet=feature_parquet,
        output_dir=tmp_path / "out",
        weights_column="uniform",
        boundary_support_column="local_image_support_score",
        boundary_floor=0.0,
        threshold=0.50,
    )

    comparison = pd.read_csv(result["comparison_path"])
    fused = tifffile.imread(tmp_path / "out" / "fused" / "fused_0000.tif")

    assert np.array_equal(fused > 0, gt > 0)
    assert comparison.loc[0, "row_family"] == "ensemble_qa"
    assert comparison.loc[0, "method_key"] == "ensemble_qa_boundary_object_vote"
    assert float(comparison.loc[0, "mean_iou"]) == 1.0


def test_boundary_aware_object_vote_accepts_per_candidate_floor(
    tmp_path: Path,
) -> None:
    gt = np.zeros((5, 5), dtype=np.uint8)
    gt[1:4, 1:4] = 1
    oversegmented = np.ones((5, 5), dtype=np.uint8)

    good_stack = tmp_path / "good.tif"
    over_stack = tmp_path / "over.tif"
    _write_stack(good_stack, gt, gt)
    _write_stack(over_stack, oversegmented, gt)

    parquet = tmp_path / "fold-1_paper_ready.parquet"
    pd.DataFrame(
        [
            {
                "cell_id": "good",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "good",
                "stacked_path": str(good_stack),
                "split": "test",
            },
            {
                "cell_id": "over",
                "campaign_number": "01",
                "original_image_key": "t0001",
                "label": 1,
                "competitor": "over",
                "stacked_path": str(over_stack),
                "split": "test",
            },
        ]
    ).to_parquet(parquet, index=False)
    feature_parquet = tmp_path / "features.parquet"
    pd.DataFrame(
        [
            {
                "candidate_id": "good",
                "local_image_support_score": 1.0,
                "source_boundary_floor": 1.0,
            },
            {
                "candidate_id": "over",
                "local_image_support_score": 0.0,
                "source_boundary_floor": 0.0,
            },
        ]
    ).to_parquet(feature_parquet, index=False)

    result = run_qa_weighted_object_vote(
        candidate_parquet=parquet,
        feature_parquet=feature_parquet,
        output_dir=tmp_path / "out",
        weights_column="uniform",
        boundary_support_column="local_image_support_score",
        boundary_floor=1.0,
        boundary_floor_column="source_boundary_floor",
        threshold=0.50,
    )

    comparison = pd.read_csv(result["comparison_path"])
    fused = tifffile.imread(tmp_path / "out" / "fused" / "fused_0000.tif")

    assert np.array_equal(fused > 0, gt > 0)
    assert comparison.loc[0, "boundary_floor_column"] == "source_boundary_floor"
    assert float(comparison.loc[0, "mean_iou"]) == 1.0


def test_discovered_object_vote_jobs_attach_feature_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = (
        tmp_path
        / "ranker_pruned_fusion_probe"
        / "Toy"
        / "sz64"
        / "baseline"
        / "fold-1"
        / "source_balanced_topk3_simple"
        / "fold-1_paper_ready_source_balanced_topk.parquet"
    )
    feature_root = tmp_path / "features"
    feature = (
        feature_root
        / "Toy"
        / "sz64"
        / "baseline"
        / "fold-1_paper_ready_pairwise_ranked.parquet"
    )
    feature.parent.mkdir(parents=True)
    feature.write_text("placeholder", encoding="utf-8")

    fake_module = types.ModuleType("scripts.run_ranker_pruned_non_simple_sweep")
    fake_module.discover_jobs = lambda phase1_root, best_by_dataset: [
        {"filtered_parquet": candidate, "policy": "source_balanced_topk3_simple"}
    ]
    monkeypatch.setitem(
        sys.modules,
        "scripts.run_ranker_pruned_non_simple_sweep",
        fake_module,
    )

    jobs = _discovered_jobs(
        tmp_path / "ranker_pruned_fusion_probe",
        tmp_path / "best.csv",
        feature_root=feature_root,
    )

    assert jobs == [
        {
            "candidate_parquet": candidate,
            "policy": "source_balanced_topk3_simple",
            "feature_parquet": feature,
        }
    ]
