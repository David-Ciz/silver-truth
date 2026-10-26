from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from silver_truth.fusion.crops_experiment import run_crops_fusion_experiment


PRUNING_POLICIES = (
    "top1_per_cluster",
    "top2_per_cluster",
    "margin_pruned",
    "source_balanced_topk",
    "fusion_safe_pruning",
)

DEFAULT_SCORE_COLUMNS = (
    "soft_hybrid_score",
    "pairwise_cluster_score",
    "candidate_ranker_score",
    "soft_qa",
    "qa_score",
    "fusion_support_rank_in_image",
    "fusion_support",
)

DEFAULT_CLUSTER_COLUMNS = (
    "ranker_cluster_id",
    "cluster_id",
    "synchronized_cluster_id",
)

DEFAULT_SOURCE_COLUMNS = (
    "candidate_source",
    "candidate_source_raw",
    "competitor",
)

DEFAULT_SUPPORT_COLUMNS = (
    "fusion_support",
    "soft_fusion_support",
    "fusion_support_rank_in_image",
)


@dataclass(frozen=True)
class RankerPruningConfig:
    policy: str = "top2_per_cluster"
    score_column: str | None = None
    cluster_column: str | None = None
    source_column: str | None = None
    support_column: str | None = None
    top_k: int = 2
    source_top_k: int = 1
    margin: float = 0.05
    fusion_safe_margin: float = 0.15
    fusion_safe_support_quantile: float = 0.50
    reindex_competitors_as_rank_slots: bool = True


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_ranker_pruned_candidate_table(
    candidates: pd.DataFrame,
    ranked_or_selected: pd.DataFrame,
    *,
    config: RankerPruningConfig = RankerPruningConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a paper-ready-compatible candidate table after ranker pruning.

    The original candidate parquet identifies rows by ``cell_id`` while ranked
    and selected soft-hybrid artifacts use ``candidate_id``. This function keeps
    the original candidate columns and appends ranker metadata only as
    additional columns, so the result can be passed directly to crop-fusion
    tooling.
    """

    if config.policy not in PRUNING_POLICIES:
        raise ValueError(
            f"Unsupported pruning policy {config.policy!r}. "
            f"Expected one of: {', '.join(PRUNING_POLICIES)}"
        )
    if "cell_id" not in candidates.columns:
        raise ValueError("Candidate parquet must include a 'cell_id' column.")
    if "candidate_id" not in ranked_or_selected.columns:
        raise ValueError("Ranked/selected table must include a 'candidate_id' column.")

    score_column = _resolve_column(
        ranked_or_selected,
        requested=config.score_column,
        candidates=DEFAULT_SCORE_COLUMNS,
        purpose="score",
    )
    cluster_column = _resolve_optional_column(
        ranked_or_selected,
        requested=config.cluster_column,
        candidates=DEFAULT_CLUSTER_COLUMNS,
    )
    source_column = _resolve_optional_column(
        ranked_or_selected,
        requested=config.source_column,
        candidates=DEFAULT_SOURCE_COLUMNS,
    )
    support_column = _resolve_optional_column(
        ranked_or_selected,
        requested=config.support_column,
        candidates=DEFAULT_SUPPORT_COLUMNS,
    )

    ranked = _deduplicate_ranked_candidates(
        ranked_or_selected,
        score_column=score_column,
    )
    ranked["_ranker_pruning_cluster"] = _cluster_values(ranked, cluster_column)
    ranked["_ranker_pruning_score"] = pd.to_numeric(
        ranked[score_column], errors="coerce"
    ).fillna(-np.inf)

    selected_ids = _select_candidate_ids(
        ranked,
        config=config,
        score_column=score_column,
        source_column=source_column,
        support_column=support_column,
    )
    pruned_ranked = ranked[ranked["candidate_id"].astype(str).isin(selected_ids)].copy()

    filtered = _merge_ranker_metadata(candidates, pruned_ranked)
    if filtered.empty:
        raise ValueError("No ranked candidate IDs matched the candidate parquet.")
    if config.reindex_competitors_as_rank_slots:
        filtered = reindex_competitors_as_rank_slots(filtered)

    stats = summarize_pruning(
        candidates=candidates,
        ranked=ranked,
        filtered=filtered,
        selected_ids=selected_ids,
        config=config,
        score_column=score_column,
        cluster_column=cluster_column,
        source_column=source_column,
        support_column=support_column,
    )
    return filtered, stats


def write_ranker_pruned_candidate_parquet(
    *,
    candidate_parquet: Path,
    ranked_candidates: Path,
    output_parquet: Path,
    config: RankerPruningConfig = RankerPruningConfig(),
    summary_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_parquet(candidate_parquet)
    ranked = load_table(ranked_candidates)
    filtered, stats = build_ranker_pruned_candidate_table(
        candidates,
        ranked,
        config=config,
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_parquet, index=False)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(summary_path, index=False)
    return filtered, stats


def run_ranker_pruned_fusion(
    *,
    candidate_parquet: Path,
    ranked_candidates: Path,
    output_dir: Path,
    config: RankerPruningConfig = RankerPruningConfig(),
    models: Sequence[str] = (),
    all_models: bool = False,
    flat_models_only: bool = False,
    weights_column: str | None = None,
    num_threads: int = 4,
    chunk_size: int = 0,
    mlflow_experiment: str = "ranker-pruned-fusion-crops",
    mlflow_tracking_path: Path | str = "data/fusion_experiments/mlruns",
    skip_fusion: bool = False,
    keep_job_dir: bool = False,
    job_dir: Path | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_parquet = output_dir / f"{candidate_parquet.stem}_{config.policy}.parquet"
    pruning_summary_path = output_dir / "ranker_pruning_summary.csv"
    _, pruning_summary = write_ranker_pruned_candidate_parquet(
        candidate_parquet=candidate_parquet,
        ranked_candidates=ranked_candidates,
        output_parquet=filtered_parquet,
        config=config,
        summary_path=pruning_summary_path,
    )

    fusion_output_dir = output_dir / "fusion"
    fusion_result = run_crops_fusion_experiment(
        qa_parquet=filtered_parquet,
        output_dir=fusion_output_dir,
        models=models,
        all_models=all_models,
        flat_models_only=flat_models_only,
        weights_column=weights_column,
        num_threads=num_threads,
        chunk_size=chunk_size,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_path=mlflow_tracking_path,
        mlflow_run_name=f"ranker_pruned_{config.policy}_{candidate_parquet.stem}",
        skip_fusion=skip_fusion,
        keep_job_dir=keep_job_dir,
        job_dir=job_dir,
        debug=debug,
    )
    comparison_path = output_dir / "ranker_pruned_fusion_comparison.csv"
    comparison = build_pruned_fusion_comparison_rows(
        leaderboard_path=Path(fusion_result["leaderboard_path"]),
        candidate_parquet=candidate_parquet,
        config=config,
        pruning_summary=pruning_summary,
    )
    comparison.to_csv(comparison_path, index=False)

    return {
        **fusion_result,
        "filtered_parquet": str(filtered_parquet),
        "pruning_summary_path": str(pruning_summary_path),
        "comparison_path": str(comparison_path),
    }


def build_pruned_fusion_comparison_rows(
    *,
    leaderboard_path: Path,
    candidate_parquet: Path,
    config: RankerPruningConfig,
    pruning_summary: pd.DataFrame,
) -> pd.DataFrame:
    leaderboard = pd.read_csv(leaderboard_path)
    metadata = infer_paper_input_metadata(candidate_parquet)
    stats = pruning_summary.iloc[0].to_dict() if not pruning_summary.empty else {}

    records: list[dict[str, Any]] = []
    for row in leaderboard.to_dict("records"):
        model = str(row.get("model", "unknown")).lower()
        mean_iou = _first_finite(
            row,
            [
                "eval_test_mean_jaccard",
                "selection_mean_jaccard",
                "eval_overall_mean_jaccard",
                "mean_jaccard",
            ],
        )
        mean_f1 = _first_finite(
            row,
            [
                "eval_test_mean_f1",
                "selection_mean_f1",
                "eval_overall_mean_f1",
                "mean_f1",
            ],
        )
        records.append(
            {
                **metadata,
                "row_family": "full_pipeline",
                "method_key": (
                    f"full_pipeline__qa_pruned_{model}__"
                    f"{_pruning_config_method_key(config)}"
                ),
                "method_label": (
                    f"QA-pruned {str(row.get('model', 'fusion'))}"
                    f"{_pruning_config_label_suffix(config)}"
                ),
                "policy": config.policy,
                "fusion_backend": row.get("model"),
                "score_column": config.score_column,
                "top_k": config.top_k,
                "source_top_k": config.source_top_k,
                "margin": config.margin,
                "fusion_safe_margin": config.fusion_safe_margin,
                "fusion_safe_support_quantile": config.fusion_safe_support_quantile,
                "mean_iou": mean_iou,
                "mean_f1": mean_f1,
                "selection_split": row.get("selection_split"),
                "fusion_success": row.get("fusion_success"),
                "filtered_candidate_rows": stats.get("filtered_candidate_rows"),
                "input_candidate_rows": stats.get("input_candidate_rows"),
                "ranked_candidate_rows": stats.get("ranked_candidate_rows"),
                "matched_ranked_rows": stats.get("matched_ranked_rows"),
                "clusters_before": stats.get("clusters_before"),
                "clusters_after": stats.get("clusters_after"),
            }
        )
    return pd.DataFrame.from_records(records)


def _pruning_config_method_key(config: RankerPruningConfig) -> str:
    parts = [config.policy]
    default = RankerPruningConfig(policy=config.policy)
    if config.score_column:
        parts.append(f"score_{_slug(config.score_column)}")
    if config.policy in {"top2_per_cluster", "source_balanced_topk"}:
        if config.top_k != default.top_k:
            parts.append(f"topk{config.top_k}")
    if config.policy == "source_balanced_topk":
        if config.source_top_k != default.source_top_k:
            parts.append(f"sourcetopk{config.source_top_k}")
    if config.policy in {"margin_pruned", "fusion_safe_pruning"}:
        if config.margin != default.margin:
            parts.append(f"margin{_float_tag(config.margin)}")
    if config.policy == "fusion_safe_pruning":
        if config.fusion_safe_margin != default.fusion_safe_margin:
            parts.append(f"safe{_float_tag(config.fusion_safe_margin)}")
        if config.fusion_safe_support_quantile != default.fusion_safe_support_quantile:
            parts.append(f"supportq{_float_tag(config.fusion_safe_support_quantile)}")
    return "__".join(parts)


def _pruning_config_label_suffix(config: RankerPruningConfig) -> str:
    details = []
    if config.score_column:
        details.append(f"score={config.score_column}")
    if config.policy in {"top2_per_cluster", "source_balanced_topk"}:
        details.append(f"top_k={config.top_k}")
    if not details:
        return ""
    return " (" + ", ".join(details) + ")"


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def build_ranker_pruned_fusion_context_table(
    *,
    pruned_rows: pd.DataFrame,
    paper_summary: pd.DataFrame,
    soft_hybrid_selection: pd.DataFrame | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in pruned_rows.to_dict("records"):
        key = _comparison_key(row)
        records.append(_context_record_from_pruned(row))

        paper_match = _match_row(paper_summary, key, row_type="fold")
        if paper_match is not None:
            records.extend(_context_records_from_paper_summary(paper_match, key))

        if soft_hybrid_selection is not None:
            soft_match = _match_row(soft_hybrid_selection, key, row_type=None)
            if soft_match is not None:
                records.append(_context_record_from_soft_hybrid(soft_match, key))

    fold_table = pd.DataFrame.from_records(records)
    if fold_table.empty:
        return fold_table
    fold_table = fold_table.drop_duplicates().reset_index(drop=True)

    mean_table = _mean_context_rows(fold_table)
    return pd.concat([fold_table, mean_table], ignore_index=True)


def write_ranker_pruned_fusion_context_table(
    *,
    comparison_paths: Sequence[Path],
    paper_summary_path: Path,
    output_path: Path,
    soft_hybrid_selection_path: Path | None = None,
) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in comparison_paths]
    if not frames:
        raise ValueError("No comparison CSVs were provided.")
    pruned_rows = pd.concat(frames, ignore_index=True)
    paper_summary = pd.read_csv(paper_summary_path)
    soft_hybrid_selection = (
        pd.read_csv(soft_hybrid_selection_path)
        if soft_hybrid_selection_path is not None
        else None
    )
    context = build_ranker_pruned_fusion_context_table(
        pruned_rows=pruned_rows,
        paper_summary=paper_summary,
        soft_hybrid_selection=soft_hybrid_selection,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context.to_csv(output_path, index=False)
    return context


def infer_paper_input_metadata(path: Path) -> dict[str, str]:
    metadata = {
        "dataset": "",
        "crop_tag": "",
        "variant": "",
        "fold": path.stem.split("_", 1)[0],
        "input_path": str(path),
    }
    parts = path.parts
    if "paper_inputs" not in parts:
        return metadata
    index = parts.index("paper_inputs")
    if len(parts) > index + 1:
        metadata["dataset"] = parts[index + 1]
    if len(parts) > index + 2:
        metadata["crop_tag"] = parts[index + 2]
    if len(parts) > index + 3:
        metadata["variant"] = parts[index + 3]
    return metadata


def summarize_pruning(
    *,
    candidates: pd.DataFrame,
    ranked: pd.DataFrame,
    filtered: pd.DataFrame,
    selected_ids: set[str],
    config: RankerPruningConfig,
    score_column: str,
    cluster_column: str | None,
    source_column: str | None,
    support_column: str | None,
) -> pd.DataFrame:
    matched_ids = set(candidates["cell_id"].astype(str)).intersection(
        set(ranked["candidate_id"].astype(str))
    )
    filtered_clusters = (
        filtered["_ranker_pruning_cluster"].nunique(dropna=False)
        if "_ranker_pruning_cluster" in filtered.columns
        else np.nan
    )
    record = {
        "policy": config.policy,
        "score_column": score_column,
        "cluster_column": cluster_column or "",
        "source_column": source_column or "",
        "support_column": support_column or "",
        "input_candidate_rows": int(len(candidates)),
        "ranked_candidate_rows": int(len(ranked)),
        "matched_ranked_rows": int(len(matched_ids)),
        "selected_ranked_rows": int(len(selected_ids)),
        "filtered_candidate_rows": int(len(filtered)),
        "input_logical_cells": _logical_cell_count(candidates),
        "filtered_logical_cells": _logical_cell_count(filtered),
        "clusters_before": int(ranked["_ranker_pruning_cluster"].nunique(dropna=False)),
        "clusters_after": filtered_clusters,
        "retained_candidate_fraction": float(len(filtered) / len(candidates))
        if len(candidates)
        else np.nan,
        "retained_matched_fraction": float(len(filtered) / len(matched_ids))
        if matched_ids
        else np.nan,
        "top_k": config.top_k,
        "source_top_k": config.source_top_k,
        "margin": config.margin,
        "fusion_safe_margin": config.fusion_safe_margin,
        "fusion_safe_support_quantile": config.fusion_safe_support_quantile,
        "reindex_competitors_as_rank_slots": config.reindex_competitors_as_rank_slots,
    }
    return pd.DataFrame([record])


def reindex_competitors_as_rank_slots(filtered: pd.DataFrame) -> pd.DataFrame:
    """Replace source competitor labels with dense rank-slot labels for fusion.

    Crop fusion uses one fixed directory per ``competitor`` value. After pruning,
    leaving the original source labels can create many all-zero missing-source
    masks per cell. Dense rank slots represent "best kept candidate",
    "second-best kept candidate", etc., which is the intended pruned candidate
    pool for fusion.
    """

    if "competitor" not in filtered.columns:
        return filtered

    working = filtered.copy()
    if "original_competitor" not in working.columns:
        working["original_competitor"] = working["competitor"]

    group_columns = [
        column
        for column in ["campaign_number", "original_image_key", "label"]
        if column in working.columns
    ]
    if len(group_columns) < 3 and "_ranker_pruning_cluster" in working.columns:
        group_columns = ["_ranker_pruning_cluster"]
    if not group_columns:
        group_columns = ["cell_id"]

    ordered = _sort_by_score(working, "_ranker_pruning_score")
    slot_index = ordered.groupby(group_columns, dropna=False).cumcount() + 1
    working.loc[ordered.index, "competitor"] = [
        f"ranker_pruned_slot_{index:02d}" for index in slot_index
    ]
    return working


def _select_candidate_ids(
    ranked: pd.DataFrame,
    *,
    config: RankerPruningConfig,
    score_column: str,
    source_column: str | None,
    support_column: str | None,
) -> set[str]:
    selected: list[pd.DataFrame] = []
    for _, group in ranked.groupby("_ranker_pruning_cluster", dropna=False, sort=False):
        ordered = _sort_by_score(group, score_column)
        if config.policy == "top1_per_cluster":
            selected.append(ordered.head(1))
        elif config.policy == "top2_per_cluster":
            selected.append(_drop_duplicate_sources(ordered, source_column).head(2))
        elif config.policy == "margin_pruned":
            selected.append(_margin_pruned_group(ordered, config.margin))
        elif config.policy == "source_balanced_topk":
            selected.append(
                _source_balanced_topk_group(
                    ordered,
                    source_column=source_column,
                    top_k=config.top_k,
                    source_top_k=config.source_top_k,
                )
            )
        elif config.policy == "fusion_safe_pruning":
            selected.append(
                _fusion_safe_pruned_group(
                    ordered,
                    support_column=support_column,
                    margin=config.margin,
                    fusion_safe_margin=config.fusion_safe_margin,
                    support_quantile=config.fusion_safe_support_quantile,
                )
            )

    if not selected:
        return set()
    selected_df = pd.concat(selected, ignore_index=False)
    return set(selected_df["candidate_id"].astype(str))


def _deduplicate_ranked_candidates(
    ranked: pd.DataFrame,
    *,
    score_column: str,
) -> pd.DataFrame:
    working = ranked.copy()
    working["_ranker_pruning_score"] = pd.to_numeric(
        working[score_column], errors="coerce"
    ).fillna(-np.inf)
    return (
        working.sort_values("_ranker_pruning_score", ascending=False)
        .drop_duplicates(subset=["candidate_id"], keep="first")
        .drop(columns=["_ranker_pruning_score"])
        .reset_index(drop=True)
    )


def _merge_ranker_metadata(
    candidates: pd.DataFrame,
    selected_ranked: pd.DataFrame,
) -> pd.DataFrame:
    candidates_with_key = candidates.copy()
    candidates_with_key["_ranker_pruning_cell_id"] = candidates_with_key[
        "cell_id"
    ].astype(str)

    metadata = selected_ranked.copy()
    metadata["_ranker_pruning_cell_id"] = metadata["candidate_id"].astype(str)
    extra_columns = [
        column
        for column in metadata.columns
        if (
            column != "_ranker_pruning_cell_id"
            and (column not in candidates.columns or column == "candidate_id")
        )
    ]
    metadata = metadata[["_ranker_pruning_cell_id", *extra_columns]]
    merged = candidates_with_key.merge(
        metadata,
        on="_ranker_pruning_cell_id",
        how="inner",
    )
    return merged.drop(columns=["_ranker_pruning_cell_id"])


def _margin_pruned_group(group: pd.DataFrame, margin: float) -> pd.DataFrame:
    if group.empty:
        return group
    best = float(group["_ranker_pruning_score"].iloc[0])
    kept = group[group["_ranker_pruning_score"] >= best - margin]
    return kept if not kept.empty else group.head(1)


def _source_balanced_topk_group(
    group: pd.DataFrame,
    *,
    source_column: str | None,
    top_k: int,
    source_top_k: int,
) -> pd.DataFrame:
    if source_column is None or source_column not in group.columns:
        return group.head(max(1, top_k))

    per_source = (
        group.groupby(source_column, dropna=False, sort=False)
        .head(max(1, source_top_k))
        .copy()
    )
    return _sort_by_score(per_source, "_ranker_pruning_score").head(max(1, top_k))


def _fusion_safe_pruned_group(
    group: pd.DataFrame,
    *,
    support_column: str | None,
    margin: float,
    fusion_safe_margin: float,
    support_quantile: float,
) -> pd.DataFrame:
    if group.empty:
        return group

    best = float(group["_ranker_pruning_score"].iloc[0])
    keep = group["_ranker_pruning_score"] >= best - margin
    if support_column is not None and support_column in group.columns:
        support = pd.to_numeric(group[support_column], errors="coerce")
        threshold = support.quantile(support_quantile)
        supported = support >= threshold
        not_strongly_rejected = group["_ranker_pruning_score"] >= (
            best - fusion_safe_margin
        )
        keep = keep | (supported & not_strongly_rejected)

    kept = group[keep]
    return kept if not kept.empty else group.head(1)


def _drop_duplicate_sources(
    group: pd.DataFrame,
    source_column: str | None,
) -> pd.DataFrame:
    if source_column is None or source_column not in group.columns:
        return group
    return group.drop_duplicates(subset=[source_column], keep="first")


def _sort_by_score(group: pd.DataFrame, score_column: str) -> pd.DataFrame:
    tie_columns = [
        column
        for column in DEFAULT_SCORE_COLUMNS
        if column in group.columns and column != score_column
    ]
    working = group.copy()
    if score_column != "_ranker_pruning_score":
        working["_ranker_pruning_score"] = pd.to_numeric(
            working[score_column], errors="coerce"
        ).fillna(-np.inf)
    sort_columns = ["_ranker_pruning_score", *tie_columns, "candidate_id"]
    ascending = [False] * (len(sort_columns) - 1) + [True]
    return working.sort_values(sort_columns, ascending=ascending, na_position="last")


def _cluster_values(ranked: pd.DataFrame, cluster_column: str | None) -> pd.Series:
    if cluster_column is not None and cluster_column in ranked.columns:
        return ranked[cluster_column].astype(str)
    if {"image_id", "label"}.issubset(ranked.columns):
        return ranked["image_id"].astype(str) + "__label_" + ranked["label"].astype(str)
    if "image_id" in ranked.columns:
        return ranked["image_id"].astype(str) + "__candidate_" + ranked[
            "candidate_id"
        ].astype(str)
    return ranked["candidate_id"].astype(str)


def _resolve_column(
    table: pd.DataFrame,
    *,
    requested: str | None,
    candidates: Iterable[str],
    purpose: str,
) -> str:
    resolved = _resolve_optional_column(table, requested=requested, candidates=candidates)
    if resolved is None:
        raise ValueError(
            f"Could not resolve {purpose} column. Tried: {', '.join(candidates)}"
        )
    return resolved


def _resolve_optional_column(
    table: pd.DataFrame,
    *,
    requested: str | None,
    candidates: Iterable[str],
) -> str | None:
    if requested:
        if requested not in table.columns:
            raise ValueError(f"Requested column {requested!r} is not present.")
        return requested
    for column in candidates:
        if column in table.columns:
            return column
    return None


def _logical_cell_count(table: pd.DataFrame) -> int | float:
    columns = ["campaign_number", "original_image_key", "label"]
    if not set(columns).issubset(table.columns):
        return np.nan
    return int(table[columns].drop_duplicates().shape[0])


def _first_finite(record: dict[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            return float(value)
    return float("nan")


def _comparison_key(row: dict[str, Any]) -> dict[str, str]:
    return {
        "dataset": str(row.get("dataset", "")),
        "crop_tag": str(row.get("crop_tag", "")),
        "variant": str(row.get("variant", "")),
        "fold": str(row.get("fold", "")),
    }


def _match_row(
    table: pd.DataFrame,
    key: dict[str, str],
    *,
    row_type: str | None,
) -> dict[str, Any] | None:
    mask = pd.Series(True, index=table.index)
    for column, value in key.items():
        if column not in table.columns or not value:
            continue
        mask &= table[column].astype(str).eq(value)
    if row_type is not None and "row_type" in table.columns:
        mask &= table["row_type"].astype(str).eq(row_type)
    matches = table[mask]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _context_record_base(key: dict[str, str]) -> dict[str, Any]:
    return {
        **key,
        "row_type": "fold",
        "n_folds": 1,
        "mean_iou": np.nan,
        "mean_f1": np.nan,
        "policy": "",
        "fusion_backend": "",
        "source_method": "",
    }


def _context_record_from_pruned(row: dict[str, Any]) -> dict[str, Any]:
    record = _context_record_base(_comparison_key(row))
    record.update(
        {
            "row_family": "full_pipeline",
            "method_key": row.get("method_key"),
            "method_label": row.get("method_label"),
            "mean_iou": row.get("mean_iou"),
            "mean_f1": row.get("mean_f1"),
            "policy": row.get("policy", ""),
            "fusion_backend": row.get("fusion_backend", ""),
            "source_method": "ranker_pruned_fusion",
        }
    )
    return record


def _context_records_from_paper_summary(
    row: dict[str, Any],
    key: dict[str, str],
) -> list[dict[str, Any]]:
    records = []
    fusion = _context_record_base(key)
    fusion.update(
        {
            "row_family": "fusion_only",
            "method_key": row.get("best_fusion_method", "fusion_only__best"),
            "method_label": row.get("best_fusion_method", "fusion_only best"),
            "mean_iou": row.get("best_fusion_iou"),
            "source_method": row.get("best_fusion_method", ""),
        }
    )
    records.append(fusion)

    competitor = _context_record_base(key)
    competitor_name = row.get("best_competitor_name", "best competitor")
    competitor.update(
        {
            "row_family": "competitor",
            "method_key": "best_competitor",
            "method_label": "Best competitor",
            "mean_iou": row.get("best_competitor_iou"),
            "source_method": competitor_name,
        }
    )
    records.append(competitor)

    silver = _context_record_base(key)
    silver.update(
        {
            "row_family": "silver_truth",
            "method_key": "silver_truth",
            "method_label": "Silver truth",
            "mean_iou": row.get("silver_truth_iou"),
            "source_method": "silver_truth",
        }
    )
    records.append(silver)
    return records


def _context_record_from_soft_hybrid(
    row: dict[str, Any],
    key: dict[str, str],
) -> dict[str, Any]:
    method = row.get("selected_method", "")
    is_reporting_mode_selection = "selected_test_iou" in row
    record = _context_record_base(key)
    if is_reporting_mode_selection:
        row_family = "reporting_mode_selected"
        method_key = "reporting_mode_selected__validation_selected"
        method_label = "Validation-selected reporting mode"
        mean_iou = row.get("selected_test_iou")
    else:
        row_family = "soft_hybrid_direct"
        method_key = "soft_hybrid_direct__validation_selected"
        method_label = "Direct soft-hybrid selected"
        mean_iou = row.get("test_mean_iou")
    record.update(
        {
            "row_family": row_family,
            "method_key": method_key,
            "method_label": method_label,
            "mean_iou": mean_iou,
            "mean_f1": row.get("test_mean_f1", np.nan),
            "source_method": method,
        }
    )
    return record


def _mean_context_rows(fold_table: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "crop_tag",
        "variant",
        "row_family",
        "method_key",
        "method_label",
        "policy",
        "fusion_backend",
    ]
    present_group_columns = [column for column in group_columns if column in fold_table]
    mean_rows = (
        fold_table.groupby(present_group_columns, dropna=False, as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            source_method=("source_method", _join_unique_values),
        )
        .copy()
    )
    mean_rows["fold"] = "mean"
    mean_rows["row_type"] = "mean"
    columns = list(fold_table.columns)
    for column in columns:
        if column not in mean_rows.columns:
            mean_rows[column] = np.nan
    return mean_rows[columns]


def _join_unique_values(values: pd.Series) -> str:
    unique = [str(value) for value in values.dropna().unique() if str(value)]
    return ";".join(unique)
