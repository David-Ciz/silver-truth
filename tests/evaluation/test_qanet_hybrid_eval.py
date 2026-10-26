#!/usr/bin/env python3
"""Tests for the QANet hybrid scoring evaluation script."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Make sure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# -------------------------------------------------------------------------
# Helpers to build minimal test DataFrames
# -------------------------------------------------------------------------

def make_qanet_scored_df(n: int = 6) -> pd.DataFrame:
    """Minimal QANet scored candidates DataFrame."""
    rng = np.random.default_rng(42)
    cluster_ids = [f"img01__cluster_000{i // 3}" for i in range(n)]
    return pd.DataFrame({
        "candidate_id": [f"cand_{i:03d}" for i in range(n)],
        "image_id": "img01",
        "split": ["train", "train", "validation", "validation", "test", "test"],
        "qanet_score": rng.uniform(0.2, 0.9, n),
        "qanet_rank_in_cluster": [0.33, 0.67, 0.33, 0.67, 0.33, 0.67],
        "true_iou": rng.uniform(0.3, 0.95, n),
        "ranker_cluster_id": cluster_ids,
    })


def make_pairwise_ranked_df(n: int = 6) -> pd.DataFrame:
    """Minimal pairwise ranked parquet DataFrame."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "candidate_id": [f"cand_{i:03d}" for i in range(n)],
        "pairwise_cluster_score": rng.uniform(0.1, 0.9, n),
        "pairwise_cluster_rank": [0.25, 0.75, 0.25, 0.75, 0.25, 0.75],
    })


# -------------------------------------------------------------------------
# Import the module under test
# -------------------------------------------------------------------------

def import_hybrid_module():
    spec_path = Path(__file__).parent.parent.parent / "scripts" / "run_qanet_hybrid_eval.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_qanet_hybrid_eval", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

class TestHybridAlphaColumn:
    def test_alpha_col_zero(self):
        mod = import_hybrid_module()
        assert mod._alpha_col(0.0) == "hybrid_rank_alpha0p00"

    def test_alpha_col_half(self):
        mod = import_hybrid_module()
        assert mod._alpha_col(0.5) == "hybrid_rank_alpha0p50"

    def test_alpha_col_one(self):
        mod = import_hybrid_module()
        assert mod._alpha_col(1.0) == "hybrid_rank_alpha1p00"


class TestBuildHybridScoredCsv:
    """Test the core merge and hybrid-rank computation logic."""

    def test_full_match_all_alphas(self, tmp_path):
        mod = import_hybrid_module()
        repo = tmp_path

        # Build entry pointing to temp files
        entry = {
            "dataset": "TEST",
            "crop_tag": "sz64",
            "variant": "baseline",
            "fold": "fold-1",
            "candidate_parquet": "candidate.parquet",
            "pairwise_ranked": "pairwise.parquet",
        }

        # Create directory structure
        qanet_dir = (
            repo / mod.PROBE_20EP_ROOT / "TEST/sz64/baseline/fold-1"
        )
        qanet_dir.mkdir(parents=True)
        pw_dir = repo / mod.PAIRWISE_SCORED_ROOT
        pw_dir.mkdir(parents=True)

        # Write test files
        qanet_df = make_qanet_scored_df(6)
        qanet_df.to_csv(qanet_dir / "qanet_scored_candidates.csv", index=False)
        pw_df = make_pairwise_ranked_df(6)
        pw_df.to_parquet(pw_dir / "pairwise.parquet", index=False)

        result = mod.build_hybrid_scored_csv(repo, entry)

        assert result is not None
        assert len(result) == 6

        # All pairwise cols merged
        assert "pairwise_cluster_score" in result.columns
        assert "pairwise_cluster_rank" in result.columns

        # All alpha columns present
        for alpha in mod.HYBRID_ALPHAS:
            col = mod._alpha_col(alpha)
            assert col in result.columns, f"Missing column {col}"
            assert result[col].between(0.0, 1.0).all()

    def test_alpha_zero_equals_pairwise_rank(self, tmp_path):
        """alpha=0.0 → hybrid rank should equal pairwise_cluster_rank."""
        mod = import_hybrid_module()
        repo = tmp_path

        entry = {
            "dataset": "TEST",
            "crop_tag": "sz64",
            "variant": "baseline",
            "fold": "fold-1",
            "candidate_parquet": "candidate.parquet",
            "pairwise_ranked": "pairwise.parquet",
        }

        qanet_dir = repo / mod.PROBE_20EP_ROOT / "TEST/sz64/baseline/fold-1"
        qanet_dir.mkdir(parents=True)
        pw_dir = repo / mod.PAIRWISE_SCORED_ROOT
        pw_dir.mkdir(parents=True)

        qanet_df = make_qanet_scored_df(6)
        qanet_df.to_csv(qanet_dir / "qanet_scored_candidates.csv", index=False)
        pw_df = make_pairwise_ranked_df(6)
        pw_df.to_parquet(pw_dir / "pairwise.parquet", index=False)

        result = mod.build_hybrid_scored_csv(repo, entry)
        assert result is not None

        col_zero = mod._alpha_col(0.0)
        np.testing.assert_allclose(
            result[col_zero].values,
            result["pairwise_cluster_rank"].values,
            atol=1e-6,
        )

    def test_alpha_one_equals_qanet_rank(self, tmp_path):
        """alpha=1.0 → hybrid rank should equal qanet_rank_in_cluster."""
        mod = import_hybrid_module()
        repo = tmp_path

        entry = {
            "dataset": "TEST",
            "crop_tag": "sz64",
            "variant": "baseline",
            "fold": "fold-1",
            "candidate_parquet": "candidate.parquet",
            "pairwise_ranked": "pairwise.parquet",
        }

        qanet_dir = repo / mod.PROBE_20EP_ROOT / "TEST/sz64/baseline/fold-1"
        qanet_dir.mkdir(parents=True)
        pw_dir = repo / mod.PAIRWISE_SCORED_ROOT
        pw_dir.mkdir(parents=True)

        qanet_df = make_qanet_scored_df(6)
        qanet_df.to_csv(qanet_dir / "qanet_scored_candidates.csv", index=False)
        pw_df = make_pairwise_ranked_df(6)
        pw_df.to_parquet(pw_dir / "pairwise.parquet", index=False)

        result = mod.build_hybrid_scored_csv(repo, entry)
        assert result is not None

        col_one = mod._alpha_col(1.0)
        np.testing.assert_allclose(
            result[col_one].values,
            result["qanet_rank_in_cluster"].values,
            atol=1e-6,
        )

    def test_alpha_half_is_average(self, tmp_path):
        """alpha=0.5 → hybrid rank should equal mean of both ranks."""
        mod = import_hybrid_module()
        repo = tmp_path

        entry = {
            "dataset": "TEST",
            "crop_tag": "sz64",
            "variant": "baseline",
            "fold": "fold-1",
            "candidate_parquet": "candidate.parquet",
            "pairwise_ranked": "pairwise.parquet",
        }

        qanet_dir = repo / mod.PROBE_20EP_ROOT / "TEST/sz64/baseline/fold-1"
        qanet_dir.mkdir(parents=True)
        pw_dir = repo / mod.PAIRWISE_SCORED_ROOT
        pw_dir.mkdir(parents=True)

        qanet_df = make_qanet_scored_df(6)
        qanet_df.to_csv(qanet_dir / "qanet_scored_candidates.csv", index=False)
        pw_df = make_pairwise_ranked_df(6)
        pw_df.to_parquet(pw_dir / "pairwise.parquet", index=False)

        result = mod.build_hybrid_scored_csv(repo, entry)
        assert result is not None

        col_half = mod._alpha_col(0.5)
        expected = (
            result["qanet_rank_in_cluster"].fillna(0.5).values * 0.5
            + result["pairwise_cluster_rank"].fillna(0.5).values * 0.5
        )
        np.testing.assert_allclose(result[col_half].values, expected, atol=1e-6)

    def test_missing_qanet_csv_returns_none(self, tmp_path):
        mod = import_hybrid_module()
        entry = {
            "dataset": "MISSING",
            "crop_tag": "sz64",
            "variant": "baseline",
            "fold": "fold-1",
            "candidate_parquet": "does_not_exist.parquet",
            "pairwise_ranked": "does_not_exist.parquet",
        }
        result = mod.build_hybrid_scored_csv(tmp_path, entry)
        assert result is None

    def test_missing_pairwise_parquet_returns_none(self, tmp_path):
        mod = import_hybrid_module()
        repo = tmp_path

        entry = {
            "dataset": "TEST",
            "crop_tag": "sz64",
            "variant": "baseline",
            "fold": "fold-1",
            "candidate_parquet": "candidate.parquet",
            "pairwise_ranked": "does_not_exist.parquet",
        }

        qanet_dir = repo / mod.PROBE_20EP_ROOT / "TEST/sz64/baseline/fold-1"
        qanet_dir.mkdir(parents=True)
        qanet_df = make_qanet_scored_df(6)
        qanet_df.to_csv(qanet_dir / "qanet_scored_candidates.csv", index=False)

        result = mod.build_hybrid_scored_csv(repo, entry)
        assert result is None

    def test_no_match_returns_none(self, tmp_path):
        """When candidate_ids don't match at all, should return None."""
        mod = import_hybrid_module()
        repo = tmp_path

        entry = {
            "dataset": "TEST",
            "crop_tag": "sz64",
            "variant": "baseline",
            "fold": "fold-1",
            "candidate_parquet": "candidate.parquet",
            "pairwise_ranked": "pairwise.parquet",
        }

        qanet_dir = repo / mod.PROBE_20EP_ROOT / "TEST/sz64/baseline/fold-1"
        qanet_dir.mkdir(parents=True)
        pw_dir = repo / mod.PAIRWISE_SCORED_ROOT
        pw_dir.mkdir(parents=True)

        qanet_df = make_qanet_scored_df(6)
        qanet_df.to_csv(qanet_dir / "qanet_scored_candidates.csv", index=False)

        # Completely different candidate IDs
        pw_df = make_pairwise_ranked_df(6)
        pw_df["candidate_id"] = [f"WRONG_{i}" for i in range(6)]
        pw_df.to_parquet(pw_dir / "pairwise.parquet", index=False)

        result = mod.build_hybrid_scored_csv(repo, entry)
        assert result is None


class TestHybridAlphaValues:
    def test_hybrid_alphas_defined(self):
        mod = import_hybrid_module()
        assert 0.0 in mod.HYBRID_ALPHAS
        assert 1.0 in mod.HYBRID_ALPHAS
        assert len(mod.HYBRID_ALPHAS) >= 3

    def test_hybrid_alphas_in_range(self):
        mod = import_hybrid_module()
        for alpha in mod.HYBRID_ALPHAS:
            assert 0.0 <= alpha <= 1.0
