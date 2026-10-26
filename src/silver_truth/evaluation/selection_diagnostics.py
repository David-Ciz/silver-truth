from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_selection_diagnostics(
    features: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    split: str = "test",
    cluster_distance: float = 4.0,
) -> dict[str, pd.DataFrame]:
    selected_ids = set(selected["candidate_id"].astype(str))
    table = features[features["split"].astype(str).eq(split)].copy()
    table["selected"] = table["candidate_id"].astype(str).isin(selected_ids)
    table["center_cluster_id"] = _assign_center_clusters(
        table, max_distance=cluster_distance
    )
    _add_cluster_diagnostic_features(table)

    cluster_rows = []
    for (image_id, cluster_id), group in table.groupby(
        ["image_id", "center_cluster_id"], dropna=False, sort=False
    ):
        oracle = _best_candidate(group)
        selected_group = group[group["selected"]]
        selected_best = _best_candidate(selected_group)
        cluster_rows.append(
            {
                "image_id": image_id,
                "center_cluster_id": cluster_id,
                "split": split,
                "n_candidates": int(len(group)),
                "n_sources": int(group["candidate_source"].nunique())
                if "candidate_source" in group.columns
                else 0,
                "n_raw_sources": int(group["candidate_source_raw"].nunique())
                if "candidate_source_raw" in group.columns
                else np.nan,
                "selected_count": int(len(selected_group)),
                "oracle_candidate_id": _value(oracle, "candidate_id"),
                "oracle_source": _value(oracle, "candidate_source"),
                "oracle_raw_source": _value(oracle, "candidate_source_raw"),
                "oracle_true_iou": _numeric_value(oracle, "true_iou"),
                "oracle_fusion_support": _numeric_value(oracle, "fusion_support"),
                "oracle_fusion_support_rank": _numeric_value(
                    oracle, "fusion_support_rank_in_image"
                ),
                "oracle_ranker_score": _numeric_value(
                    oracle, "candidate_ranker_score"
                ),
                "oracle_pairwise_score": _numeric_value(
                    oracle, "pairwise_cluster_score"
                ),
                "oracle_pairwise_rank": _numeric_value(
                    oracle, "pairwise_cluster_rank"
                ),
                "oracle_soft_hybrid_score": _numeric_value(
                    oracle, "soft_hybrid_score"
                ),
                "oracle_qa_score": _numeric_value(oracle, "qa_score"),
                "oracle_qa_rank": _numeric_value(oracle, "qa_rank_in_image"),
                "oracle_temporal_support": _numeric_value(
                    oracle, "source_temporal_support_score"
                ),
                "oracle_temporal_distance": _numeric_value(
                    oracle, "source_temporal_min_center_dist_norm"
                ),
                "oracle_source_prior": _numeric_value(
                    oracle, "source_train_mean_iou"
                ),
                "selected_candidate_id": _value(selected_best, "candidate_id"),
                "selected_source": _value(selected_best, "candidate_source"),
                "selected_raw_source": _value(selected_best, "candidate_source_raw"),
                "selected_true_iou": _numeric_value(selected_best, "true_iou"),
                "selected_fusion_support": _numeric_value(
                    selected_best, "fusion_support"
                ),
                "selected_fusion_support_rank": _numeric_value(
                    selected_best, "fusion_support_rank_in_image"
                ),
                "selected_ranker_score": _numeric_value(
                    selected_best, "candidate_ranker_score"
                ),
                "selected_pairwise_score": _numeric_value(
                    selected_best, "pairwise_cluster_score"
                ),
                "selected_pairwise_rank": _numeric_value(
                    selected_best, "pairwise_cluster_rank"
                ),
                "selected_soft_hybrid_score": _numeric_value(
                    selected_best, "soft_hybrid_score"
                ),
                "selected_qa_score": _numeric_value(selected_best, "qa_score"),
                "selected_qa_rank": _numeric_value(selected_best, "qa_rank_in_image"),
                "selected_temporal_support": _numeric_value(
                    selected_best, "source_temporal_support_score"
                ),
                "selected_temporal_distance": _numeric_value(
                    selected_best, "source_temporal_min_center_dist_norm"
                ),
                "selected_source_prior": _numeric_value(
                    selected_best, "source_train_mean_iou"
                ),
                "oracle_gap": _numeric_value(oracle, "true_iou")
                - _numeric_value(selected_best, "true_iou"),
                "fusion_support_delta": _numeric_value(
                    oracle, "fusion_support"
                )
                - _numeric_value(selected_best, "fusion_support"),
                "qa_score_delta": _numeric_value(oracle, "qa_score")
                - _numeric_value(selected_best, "qa_score"),
                "pairwise_score_delta": _numeric_value(
                    oracle, "pairwise_cluster_score"
                )
                - _numeric_value(selected_best, "pairwise_cluster_score"),
                "soft_hybrid_score_delta": _numeric_value(
                    oracle, "soft_hybrid_score"
                )
                - _numeric_value(selected_best, "soft_hybrid_score"),
                "temporal_support_delta": _numeric_value(
                    oracle, "source_temporal_support_score"
                )
                - _numeric_value(selected_best, "source_temporal_support_score"),
                "source_prior_delta": _numeric_value(
                    oracle, "source_train_mean_iou"
                )
                - _numeric_value(selected_best, "source_train_mean_iou"),
                "oracle_score_disadvantage": _score_disadvantage(
                    oracle, selected_best
                ),
                "failure_mode": _failure_mode(oracle, selected_best),
                "sources": ";".join(sorted(group["candidate_source"].astype(str).unique()))
                if "candidate_source" in group.columns
                else "",
                "raw_sources": ";".join(
                    sorted(group["candidate_source_raw"].astype(str).unique())
                )
                if "candidate_source_raw" in group.columns
                else "",
            }
        )

    cluster_df = pd.DataFrame.from_records(cluster_rows)
    image_df = _summarize_image_gaps(table, cluster_df, split=split)
    return {
        "candidate_rows": table,
        "cluster_oracle_gap": cluster_df,
        "image_oracle_gap": image_df,
        "failure_mode_summary": _summarize_failure_modes(cluster_df),
    }


def check_duplicate_sources(
    features: pd.DataFrame,
    *,
    source_pairs: Iterable[tuple[str, str]],
    key_columns: tuple[str, ...] = ("image_id", "label"),
) -> pd.DataFrame:
    records = []
    numeric_columns = [
        "center_y",
        "center_x",
        "crop_y_start",
        "crop_y_end",
        "crop_x_start",
        "crop_x_end",
        "fusion_support",
        "qa_score",
        "true_iou",
    ]
    for source_a, source_b in source_pairs:
        pair = features[
            features["candidate_source_raw"].astype(str).isin([source_a, source_b])
        ].copy()
        grouped = pair.groupby(list(key_columns), dropna=False)
        paired_groups = [
            group
            for _, group in grouped
            if {source_a, source_b}.issubset(
                set(group["candidate_source_raw"].astype(str))
            )
        ]
        record: dict[str, object] = {
            "source_a": source_a,
            "source_b": source_b,
            "rows_a": int(pair["candidate_source_raw"].astype(str).eq(source_a).sum()),
            "rows_b": int(pair["candidate_source_raw"].astype(str).eq(source_b).sum()),
            "paired_groups": int(len(paired_groups)),
        }
        for column in numeric_columns:
            diffs = []
            if column not in pair.columns:
                record[f"max_abs_diff_{column}"] = np.nan
                continue
            for group in paired_groups:
                a_rows = group[group["candidate_source_raw"].astype(str).eq(source_a)]
                b_rows = group[group["candidate_source_raw"].astype(str).eq(source_b)]
                if a_rows.empty or b_rows.empty:
                    continue
                a_value = pd.to_numeric(a_rows.iloc[[0]][column], errors="coerce").iloc[
                    0
                ]
                b_value = pd.to_numeric(b_rows.iloc[[0]][column], errors="coerce").iloc[
                    0
                ]
                if pd.notna(a_value) and pd.notna(b_value):
                    diffs.append(abs(float(a_value) - float(b_value)))
            record[f"max_abs_diff_{column}"] = max(diffs) if diffs else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def write_selection_diagnostics(
    *,
    features_path: Path,
    selected_path: Path,
    output_dir: Path,
    split: str = "test",
    cluster_distance: float = 4.0,
    duplicate_source_pairs: Iterable[tuple[str, str]] = (),
) -> dict[str, Path]:
    features = load_table(features_path)
    selected = load_table(selected_path)
    diagnostics = build_selection_diagnostics(
        features, selected, split=split, cluster_distance=cluster_distance
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "candidate_rows": output_dir / f"{split}_candidate_rows.csv",
        "cluster_oracle_gap": output_dir / f"{split}_cluster_oracle_gap.csv",
        "image_oracle_gap": output_dir / f"{split}_image_oracle_gap.csv",
        "failure_mode_summary": output_dir / f"{split}_failure_mode_summary.csv",
    }
    for key, path in paths.items():
        diagnostics[key].to_csv(path, index=False)

    if duplicate_source_pairs:
        duplicate_df = check_duplicate_sources(
            features, source_pairs=duplicate_source_pairs
        )
        duplicate_path = output_dir / "duplicate_source_check.csv"
        duplicate_df.to_csv(duplicate_path, index=False)
        paths["duplicate_source_check"] = duplicate_path
    return paths


def _assign_center_clusters(
    features: pd.DataFrame, *, max_distance: float
) -> pd.Series:
    cluster_ids = pd.Series(index=features.index, dtype=object)
    for image_id, group in features.groupby("image_id", dropna=False, sort=False):
        centers = group[["center_y", "center_x"]].to_numpy(dtype=float)
        labels = _connected_components(centers, max_distance=max_distance)
        cluster_ids.loc[group.index] = [
            f"{image_id}__cluster_{label:04d}" for label in labels
        ]
    return cluster_ids


def _add_cluster_diagnostic_features(table: pd.DataFrame) -> None:
    if "source_temporal_support_score" not in table.columns:
        table["source_temporal_support_score"] = table.get("temporal_support_score")
    if "source_temporal_min_center_dist_norm" not in table.columns:
        table["source_temporal_min_center_dist_norm"] = table.get(
            "temporal_min_center_dist_norm"
        )
    if "candidate_ranker_score" not in table.columns and "pairwise_cluster_score" in table.columns:
        table["candidate_ranker_score"] = table["pairwise_cluster_score"]
    if "soft_hybrid_score" not in table.columns:
        score_source = table.get("pairwise_cluster_score", table.get("qa_score"))
        table["soft_hybrid_score"] = pd.to_numeric(score_source, errors="coerce")


def _connected_components(points: np.ndarray, *, max_distance: float) -> list[int]:
    n_points = len(points)
    parent = list(range(n_points))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(n_points):
        if np.isnan(points[left]).any():
            continue
        for right in range(left + 1, n_points):
            if np.isnan(points[right]).any():
                continue
            if float(np.linalg.norm(points[left] - points[right])) <= max_distance:
                union(left, right)

    roots: dict[int, int] = {}
    labels = []
    for index in range(n_points):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels


def _best_candidate(group: pd.DataFrame) -> pd.Series | None:
    if group.empty:
        return None
    sort_columns = [
        column
        for column in ("true_iou", "candidate_ranker_score", "qa_score")
        if column in group.columns
    ]
    if not sort_columns:
        return group.iloc[0]
    return group.sort_values(sort_columns, ascending=False, na_position="last").iloc[0]


def _summarize_image_gaps(
    candidates: pd.DataFrame, clusters: pd.DataFrame, *, split: str
) -> pd.DataFrame:
    rows = []
    for image_id, candidate_group in candidates.groupby("image_id", dropna=False):
        cluster_group = clusters[clusters["image_id"].eq(image_id)]
        selected_candidates = candidate_group[candidate_group["selected"]]
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "n_candidates": int(len(candidate_group)),
                "n_clusters": int(cluster_group["center_cluster_id"].nunique()),
                "selected_count": int(len(selected_candidates)),
                "oracle_mean_cluster_iou": float(
                    pd.to_numeric(
                        cluster_group["oracle_true_iou"], errors="coerce"
                    ).mean()
                ),
                "selected_mean_cluster_iou": float(
                    pd.to_numeric(
                        cluster_group["selected_true_iou"], errors="coerce"
                    ).mean()
                ),
                "mean_cluster_oracle_gap": float(
                    pd.to_numeric(cluster_group["oracle_gap"], errors="coerce").mean()
                ),
                "max_cluster_oracle_gap": float(
                    pd.to_numeric(cluster_group["oracle_gap"], errors="coerce").max()
                ),
                "image_max_candidate_iou": float(
                    pd.to_numeric(candidate_group["true_iou"], errors="coerce").max()
                ),
                "selected_max_candidate_iou": float(
                    pd.to_numeric(
                        selected_candidates["true_iou"], errors="coerce"
                    ).max()
                )
                if not selected_candidates.empty
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _summarize_failure_modes(clusters: pd.DataFrame) -> pd.DataFrame:
    if clusters.empty or "failure_mode" not in clusters.columns:
        return pd.DataFrame()
    rows = clusters.copy()
    rows["oracle_gap"] = pd.to_numeric(rows["oracle_gap"], errors="coerce")
    rows = rows[rows["oracle_gap"].fillna(0.0) > 0]
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "failure_mode",
                "n_clusters",
                "mean_oracle_gap",
                "sum_oracle_gap",
                "mean_oracle_score_disadvantage",
            ]
        )
    return (
        rows.groupby("failure_mode", dropna=False, as_index=False)
        .agg(
            n_clusters=("failure_mode", "size"),
            mean_oracle_gap=("oracle_gap", "mean"),
            sum_oracle_gap=("oracle_gap", "sum"),
            mean_oracle_score_disadvantage=("oracle_score_disadvantage", "mean"),
        )
        .sort_values(
            ["sum_oracle_gap", "n_clusters"], ascending=[False, False]
        )
        .reset_index(drop=True)
    )


def _score_disadvantage(
    oracle: pd.Series | None, selected: pd.Series | None
) -> float:
    score_columns = (
        "soft_hybrid_score",
        "pairwise_cluster_score",
        "candidate_ranker_score",
        "qa_score",
    )
    for column in score_columns:
        oracle_score = _numeric_value(oracle, column)
        selected_score = _numeric_value(selected, column)
        if np.isfinite(oracle_score) and np.isfinite(selected_score):
            return selected_score - oracle_score
    return float("nan")


def _failure_mode(oracle: pd.Series | None, selected: pd.Series | None) -> str:
    if oracle is None:
        return "no_oracle"
    if selected is None:
        return "missed_cluster"
    if str(_value(oracle, "candidate_id")) == str(_value(selected, "candidate_id")):
        return "selected_oracle"

    score_disadvantage = _score_disadvantage(oracle, selected)
    qa_delta = _numeric_value(oracle, "qa_score") - _numeric_value(
        selected, "qa_score"
    )
    support_delta = _numeric_value(oracle, "fusion_support") - _numeric_value(
        selected, "fusion_support"
    )
    temporal_delta = _numeric_value(
        oracle, "source_temporal_support_score"
    ) - _numeric_value(selected, "source_temporal_support_score")
    source_prior_delta = _numeric_value(
        oracle, "source_train_mean_iou"
    ) - _numeric_value(selected, "source_train_mean_iou")

    if np.isfinite(score_disadvantage) and score_disadvantage > 0:
        return "oracle_underranked"
    if np.isfinite(qa_delta) and qa_delta < 0:
        return "oracle_lower_qa"
    if np.isfinite(support_delta) and support_delta < 0:
        return "oracle_lower_support"
    if np.isfinite(temporal_delta) and temporal_delta < 0:
        return "oracle_lower_temporal_support"
    if np.isfinite(source_prior_delta) and source_prior_delta < 0:
        return "oracle_lower_source_prior"
    return "selection_tiebreak_or_suppression"


def _value(row: pd.Series | None, column: str) -> object:
    if row is None or column not in row:
        return np.nan
    return row[column]


def _numeric_value(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return float("nan")
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write candidate selection oracle-gap diagnostics."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cluster-distance", type=float, default=4.0)
    parser.add_argument(
        "--duplicate-source",
        nargs=2,
        action="append",
        metavar=("SOURCE_A", "SOURCE_B"),
        default=[],
    )
    args = parser.parse_args()

    paths = write_selection_diagnostics(
        features_path=args.features,
        selected_path=args.selected,
        output_dir=args.output_dir,
        split=args.split,
        cluster_distance=args.cluster_distance,
        duplicate_source_pairs=[tuple(pair) for pair in args.duplicate_source],
    )
    for key, path in paths.items():
        print(f"Wrote {key}: {path}")


if __name__ == "__main__":
    main()
