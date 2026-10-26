#!/usr/bin/env python3
"""Build non-QA-pruned candidate pools for the ENSEMBLE experiments."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from silver_truth.evaluation.ranker_pruned_fusion import (  # noqa: E402
    RankerPruningConfig,
    build_ranker_pruned_candidate_table,
)
from silver_truth.evaluation.soft_hybrid import (  # noqa: E402
    SoftHybridWeights,
    score_candidates,
)


DEFAULT_FEATURE_ROOT = (
    REPO_ROOT
    / "hpc_results"
    / "soft_hybrid_probe"
    / "all_available_square_rect_full_image_local_support_pairwise_source_trainonly"
    / "features"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "hpc_results"
    / "soft_hybrid_probe"
    / "non_qa_pruned_ensemble"
)
DEFAULT_CANDIDATE_RUN = "noqa_soft_hybrid_source_balanced_top3"
DEFAULT_PAIRWISE_CANDIDATE_RUN = "pairwise_context_source_balanced_top3"
DEFAULT_PAIRWISE_QA_BLEND_PREFIX = "pairwise_qa_blend"
NO_QA_WEIGHTS = SoftHybridWeights(
    fusion_support=0.55,
    qa=0.00,
    temporal=0.20,
    morphology=0.15,
    source_prior=0.05,
    local_image_support=0.05,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-run", default=None)
    parser.add_argument(
        "--score-mode",
        choices=("noqa_soft_hybrid", "pairwise_context", "pairwise_qa_blend"),
        default="noqa_soft_hybrid",
        help=(
            "Pruning score to use. noqa_soft_hybrid computes soft_hybrid_score "
            "with QA weight zero; pairwise_context uses pairwise_cluster_score; "
            "pairwise_qa_blend combines pairwise_cluster_score with qa_score."
        ),
    )
    parser.add_argument(
        "--qa-blend-alpha",
        type=float,
        default=0.05,
        help=(
            "Direct QA weight for --score-mode pairwise_qa_blend. The blended "
            "score is (1-alpha)*pairwise_cluster_score + alpha*qa_score."
        ),
    )
    parser.add_argument("--pattern", default="**/*_paper_ready_pairwise_ranked.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    candidate_run = args.candidate_run or _default_candidate_run(
        args.score_mode, args.qa_blend_alpha
    )

    feature_root = _resolve_path(args.feature_root)
    output_root = _resolve_path(args.output_root)
    feature_paths = sorted(feature_root.glob(args.pattern))
    if args.limit is not None:
        feature_paths = feature_paths[: args.limit]
    if not feature_paths:
        raise SystemExit(f"No ranked feature parquets found under {feature_root}")

    score_label = _score_label(args.score_mode, args.qa_blend_alpha)
    candidate_root = output_root / "candidate_pools"
    scored_root = output_root / f"scored_{score_label}_features"
    summary_rows: list[pd.DataFrame] = []

    config = RankerPruningConfig(
        policy="source_balanced_topk",
        score_column="soft_hybrid_score",
        cluster_column="ranker_cluster_id",
        source_column="candidate_source",
        top_k=3,
        source_top_k=1,
        reindex_competitors_as_rank_slots=True,
    )

    for feature_path in feature_paths:
        job = _parse_feature_job(feature_root, feature_path)
        output_parquet = (
            candidate_root
            / job["dataset"]
            / job["crop_tag"]
            / job["variant"]
            / job["fold"]
            / config.policy
            / candidate_run
            / "prepared_soft_gate_candidates.parquet"
        )
        scored_path = (
            scored_root
            / job["dataset"]
            / job["crop_tag"]
            / job["variant"]
            / f"{job['fold']}_{score_label}_ranked.parquet"
        )
        summary_path = output_parquet.parent / "ranker_pruning_summary.csv"
        if output_parquet.exists() and summary_path.exists() and not args.overwrite:
            summary = pd.read_csv(summary_path)
            summary_rows.append(
                _annotate_summary(
                    summary,
                    job,
                    feature_path,
                    output_parquet,
                    candidate_run,
                    args.score_mode,
                )
            )
            print(f"Using existing {output_parquet}")
            continue

        features = pd.read_parquet(feature_path)
        candidate_parquet = _candidate_parquet_from_features(features)
        candidates = pd.read_parquet(candidate_parquet)
        scored, score_column = _score_features(
            features, args.score_mode, args.qa_blend_alpha
        )
        config = RankerPruningConfig(
            policy="source_balanced_topk",
            score_column=score_column,
            cluster_column="ranker_cluster_id",
            source_column="candidate_source",
            top_k=3,
            source_top_k=1,
            reindex_competitors_as_rank_slots=True,
        )

        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(scored_path, index=False)

        filtered, summary = build_ranker_pruned_candidate_table(
            candidates,
            scored,
            config=config,
        )
        filtered.to_parquet(output_parquet, index=False)
        annotated = _annotate_summary(
            summary,
            job,
            feature_path,
            output_parquet,
            candidate_run,
            args.score_mode,
        )
        annotated["candidate_parquet"] = str(candidate_parquet)
        annotated["scored_feature_parquet"] = str(scored_path)
        annotated["soft_hybrid_weight_fusion_support"] = NO_QA_WEIGHTS.fusion_support
        annotated["soft_hybrid_weight_qa"] = NO_QA_WEIGHTS.qa
        annotated["soft_hybrid_weight_temporal"] = NO_QA_WEIGHTS.temporal
        annotated["soft_hybrid_weight_morphology"] = NO_QA_WEIGHTS.morphology
        annotated["soft_hybrid_weight_source_prior"] = NO_QA_WEIGHTS.source_prior
        annotated["soft_hybrid_weight_local_image_support"] = (
            NO_QA_WEIGHTS.local_image_support
        )
        if args.score_mode == "pairwise_qa_blend":
            annotated["qa_blend_alpha"] = args.qa_blend_alpha
        annotated.to_csv(summary_path, index=False)
        summary_rows.append(annotated)
        print(
            f"Wrote {output_parquet} "
            f"({int(summary['filtered_candidate_rows'].iloc[0])} rows)"
        )

    combined = pd.concat(summary_rows, ignore_index=True)
    output_root.mkdir(parents=True, exist_ok=True)
    combined_path = output_root / f"{score_label}_candidate_pool_summary.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Wrote combined summary: {combined_path}")


def _parse_feature_job(feature_root: Path, feature_path: Path) -> dict[str, str]:
    rel = feature_path.relative_to(feature_root)
    if len(rel.parts) < 4:
        raise ValueError(f"Unexpected feature path layout: {feature_path}")
    match = re.match(r"(fold-\d+)_paper_ready", feature_path.name)
    if not match:
        raise ValueError(f"Could not parse fold from {feature_path.name}")
    return {
        "dataset": rel.parts[0],
        "crop_tag": rel.parts[1],
        "variant": rel.parts[2],
        "fold": match.group(1),
    }


def _candidate_parquet_from_features(features: pd.DataFrame) -> Path:
    if "source_path" not in features.columns:
        raise ValueError("Ranked features must include source_path.")
    paths = [
        value
        for value in features["source_path"].dropna().astype(str).unique()
        if value.strip()
    ]
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one source_path, found {paths}")
    return _resolve_path(Path(paths[0]))


def _annotate_summary(
    summary: pd.DataFrame,
    job: dict[str, str],
    feature_path: Path,
    output_parquet: Path,
    candidate_run: str,
    score_mode: str,
) -> pd.DataFrame:
    annotated = summary.copy()
    for key, value in job.items():
        annotated[key] = value
    annotated["ranked_feature_parquet"] = str(feature_path)
    annotated["prepared_parquet"] = str(output_parquet)
    annotated["candidate_run"] = candidate_run
    annotated["experiment_family"] = "non_qa_pruned_ensemble"
    annotated["score_mode"] = score_mode
    annotated["pruning_score_definition"] = _score_definition(score_mode)
    return annotated


def _score_features(
    features: pd.DataFrame,
    score_mode: str,
    qa_blend_alpha: float,
) -> tuple[pd.DataFrame, str]:
    if score_mode == "noqa_soft_hybrid":
        return score_candidates(features, weights=NO_QA_WEIGHTS), "soft_hybrid_score"
    if score_mode == "pairwise_context":
        if "pairwise_cluster_score" not in features.columns:
            raise ValueError("pairwise_context mode requires pairwise_cluster_score.")
        return features.copy(), "pairwise_cluster_score"
    if score_mode == "pairwise_qa_blend":
        if not 0.0 <= qa_blend_alpha <= 1.0:
            raise ValueError("--qa-blend-alpha must be in [0, 1].")
        missing = [
            column
            for column in ("pairwise_cluster_score", "qa_score")
            if column not in features.columns
        ]
        if missing:
            raise ValueError(
                "pairwise_qa_blend mode requires columns: " + ", ".join(missing)
            )
        scored = features.copy()
        pairwise = pd.to_numeric(
            scored["pairwise_cluster_score"], errors="coerce"
        ).fillna(0.5)
        qa = pd.to_numeric(scored["qa_score"], errors="coerce").fillna(0.5)
        scored["pairwise_qa_blend_score"] = (
            (1.0 - qa_blend_alpha) * pairwise.clip(0.0, 1.0)
            + qa_blend_alpha * qa.clip(0.0, 1.0)
        )
        return scored, "pairwise_qa_blend_score"
    raise ValueError(f"Unsupported score mode: {score_mode}")


def _score_definition(score_mode: str) -> str:
    if score_mode == "noqa_soft_hybrid":
        return "soft_hybrid_score with qa weight set to 0.0"
    if score_mode == "pairwise_context":
        return "pairwise_cluster_score"
    if score_mode == "pairwise_qa_blend":
        return "pairwise_qa_blend_score = (1-alpha)*pairwise_cluster_score + alpha*qa_score"
    raise ValueError(f"Unsupported score mode: {score_mode}")


def _default_candidate_run(score_mode: str, qa_blend_alpha: float) -> str:
    if score_mode == "noqa_soft_hybrid":
        return DEFAULT_CANDIDATE_RUN
    if score_mode == "pairwise_context":
        return DEFAULT_PAIRWISE_CANDIDATE_RUN
    if score_mode == "pairwise_qa_blend":
        alpha = str(qa_blend_alpha).replace(".", "p")
        return f"{DEFAULT_PAIRWISE_QA_BLEND_PREFIX}_a{alpha}_source_balanced_top3"
    raise ValueError(f"Unsupported score mode: {score_mode}")


def _score_label(score_mode: str, qa_blend_alpha: float) -> str:
    if score_mode != "pairwise_qa_blend":
        return score_mode
    alpha = str(qa_blend_alpha).replace(".", "p")
    return f"{score_mode}_a{alpha}"


def _resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    main()
