#!/usr/bin/env python3
"""Summarize final MLflow evaluation runs from an exported SQLite DB."""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tabulate import tabulate


FINAL_EVAL_LEVEL = "full_image_label"
OUR_PIPELINES = {"fusion", "qa_selector", "ensemble"}


@dataclass(frozen=True)
class RunRecord:
    experiment_name: str
    run_name: str
    pipeline_family: str
    setup_name: str
    dataset: str
    split: str
    test_iou: float


@dataclass(frozen=True)
class FoldSummary:
    dataset: str
    crop_tag: str
    fold: str
    silver_truth_iou: float | None
    best_competitor_name: str | None
    best_competitor_iou: float | None
    best_fuser_name: str | None
    best_fuser_iou: float | None
    best_ours_name: str | None
    best_ours_iou: float | None


EXPERIMENT_PATTERN = re.compile(
    r"^ablation-(?P<dataset>BF-C2DL-[A-Za-z]+)-(?P<variant>.+)-(?P<fold>fold-[12]|mixed)$"
)
VARIANT_CROP_PATTERN = re.compile(r"(?P<crop>sz\d+|crop\d+)")


def _canonical_crop_tag(dataset: str, variant: str) -> str:
    match = VARIANT_CROP_PATTERN.search(variant)
    if not match:
        if dataset == "BF-C2DL-HSC" and variant == "baseline":
            return "sz64"
        return "unknown"
    crop = match.group("crop")
    if crop.startswith("crop"):
        return f"sz{crop.removeprefix('crop')}"
    return crop


def _parse_experiment_name(experiment_name: str) -> tuple[str, str, str] | None:
    match = EXPERIMENT_PATTERN.match(experiment_name)
    if not match:
        return None
    dataset = match.group("dataset")
    variant = match.group("variant")
    fold = match.group("fold")
    return dataset, _canonical_crop_tag(dataset, variant), fold


def load_final_runs(db_path: Path) -> list[RunRecord]:
    query = """
    WITH run_info AS (
        SELECT
            r.run_uuid AS run_uuid,
            e.name AS experiment_name,
            MAX(CASE WHEN t.key='mlflow.runName' THEN t.value END) AS run_name,
            MAX(CASE WHEN t.key='evaluation_level' THEN t.value END) AS evaluation_level,
            MAX(CASE WHEN t.key='pipeline_family' THEN t.value END) AS pipeline_family,
            MAX(CASE WHEN t.key='setup_name' THEN t.value END) AS setup_name,
            MAX(CASE WHEN t.key='dataset' THEN t.value END) AS dataset,
            MAX(CASE WHEN t.key='split' THEN t.value END) AS split
        FROM runs r
        JOIN experiments e ON e.experiment_id = r.experiment_id
        LEFT JOIN tags t ON t.run_uuid = r.run_uuid
        GROUP BY r.run_uuid, e.name
    )
    SELECT
        run_info.experiment_name,
        run_info.run_name,
        COALESCE(run_info.pipeline_family, '') AS pipeline_family,
        COALESCE(run_info.setup_name, '') AS setup_name,
        COALESCE(run_info.dataset, '') AS dataset,
        COALESCE(run_info.split, '') AS split,
        CAST(lm.value AS REAL) AS test_iou
    FROM run_info
    JOIN latest_metrics lm
      ON lm.run_uuid = run_info.run_uuid
     AND lm.key = 'test_iou'
    WHERE run_info.evaluation_level = ?
    """

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, (FINAL_EVAL_LEVEL,)).fetchall()

    deduped: dict[tuple[str, str], RunRecord] = {}
    for row in rows:
        record = RunRecord(*row)
        key = (record.experiment_name, record.run_name)
        existing = deduped.get(key)
        if existing is None or record.test_iou > existing.test_iou:
            deduped[key] = record
    return sorted(deduped.values(), key=lambda record: record.test_iou, reverse=True)


def _pick_best(records: Iterable[RunRecord]) -> tuple[str | None, float | None]:
    records_list = list(records)
    if not records_list:
        return None, None
    best = max(records_list, key=lambda record: record.test_iou)
    return best.run_name, best.test_iou


def _is_fusion_only_baseline(record: RunRecord) -> bool:
    return record.pipeline_family == "fusion" and record.run_name.startswith(
        "fusion_only__"
    )


def build_fold_summaries(records: Iterable[RunRecord]) -> list[FoldSummary]:
    grouped: dict[tuple[str, str, str], list[RunRecord]] = {}
    for record in records:
        parsed = _parse_experiment_name(record.experiment_name)
        if parsed is None:
            continue
        dataset, crop_tag, fold = parsed
        grouped.setdefault((dataset, crop_tag, fold), []).append(record)

    summaries: list[FoldSummary] = []
    for (dataset, crop_tag, fold), rows in sorted(grouped.items()):
        silver_truth = next(
            (record.test_iou for record in rows if record.run_name == "SILVER-TRUTH"),
            None,
        )
        competitor_name, competitor_iou = _pick_best(
            record for record in rows if record.pipeline_family == "competitor"
        )
        fuser_name, fuser_iou = _pick_best(
            record for record in rows if _is_fusion_only_baseline(record)
        )
        ours_name, ours_iou = _pick_best(
            record for record in rows if record.pipeline_family in OUR_PIPELINES
        )
        summaries.append(
            FoldSummary(
                dataset=dataset,
                crop_tag=crop_tag,
                fold=fold,
                silver_truth_iou=silver_truth,
                best_competitor_name=competitor_name,
                best_competitor_iou=competitor_iou,
                best_fuser_name=fuser_name,
                best_fuser_iou=fuser_iou,
                best_ours_name=ours_name,
                best_ours_iou=ours_iou,
            )
        )
    return summaries


def _fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def print_fold_table(summaries: list[FoldSummary]) -> None:
    score_rows: list[list[str]] = []
    name_rows: list[list[str]] = []
    for summary in summaries:
        gap_to_silver = (
            None
            if summary.best_ours_iou is None or summary.silver_truth_iou is None
            else summary.best_ours_iou - summary.silver_truth_iou
        )
        gap_to_comp = (
            None
            if summary.best_ours_iou is None or summary.best_competitor_iou is None
            else summary.best_ours_iou - summary.best_competitor_iou
        )
        gap_to_fuser = (
            None
            if summary.best_ours_iou is None or summary.best_fuser_iou is None
            else summary.best_ours_iou - summary.best_fuser_iou
        )
        score_rows.append(
            [
                summary.dataset,
                summary.crop_tag,
                summary.fold,
                _fmt_float(summary.silver_truth_iou),
                _fmt_float(summary.best_competitor_iou),
                _fmt_float(summary.best_fuser_iou),
                _fmt_float(summary.best_ours_iou),
                _fmt_float(gap_to_fuser),
                _fmt_float(gap_to_silver),
                _fmt_float(gap_to_comp),
            ]
        )
        name_rows.append(
            [
                summary.dataset,
                summary.crop_tag,
                summary.fold,
                summary.best_competitor_name or "",
                summary.best_fuser_name or "",
                summary.best_ours_name or "",
            ]
        )

    print("Per-fold scores")
    print(
        tabulate(
            score_rows,
            headers=[
                "dataset",
                "crop",
                "fold",
                "silver",
                "comp_iou",
                "fuser_iou",
                "our_iou",
                "gap_fuser",
                "gap_silver",
                "gap_comp",
            ],
            tablefmt="github",
        )
    )
    print("\nPer-fold methods")
    print(
        tabulate(
            name_rows,
            headers=[
                "dataset",
                "crop",
                "fold",
                "best_competitor",
                "best_fuser",
                "best_ours",
            ],
            tablefmt="github",
        )
    )


def print_crop_average_table(
    summaries: list[FoldSummary], records: Iterable[RunRecord]
) -> None:
    grouped: dict[tuple[str, str], list[FoldSummary]] = {}
    for summary in summaries:
        grouped.setdefault((summary.dataset, summary.crop_tag), []).append(summary)

    record_groups: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        parsed = _parse_experiment_name(record.experiment_name)
        if parsed is None:
            continue
        dataset, crop_tag, _fold = parsed
        record_groups.setdefault((dataset, crop_tag), []).append(record)

    score_rows: list[list[str]] = []
    name_rows: list[list[str]] = []
    for (dataset, crop_tag), fold_rows in sorted(grouped.items()):
        silver_mean = sum(
            row.silver_truth_iou for row in fold_rows if row.silver_truth_iou is not None
        ) / max(1, sum(1 for row in fold_rows if row.silver_truth_iou is not None))
        comp_mean = sum(
            row.best_competitor_iou
            for row in fold_rows
            if row.best_competitor_iou is not None
        ) / max(1, sum(1 for row in fold_rows if row.best_competitor_iou is not None))
        fuser_mean = sum(
            row.best_fuser_iou for row in fold_rows if row.best_fuser_iou is not None
        ) / max(1, sum(1 for row in fold_rows if row.best_fuser_iou is not None))
        oracle_ours_mean = sum(
            row.best_ours_iou for row in fold_rows if row.best_ours_iou is not None
        ) / max(1, sum(1 for row in fold_rows if row.best_ours_iou is not None))

        best_single_method_name = ""
        best_single_method_mean = None
        best_fuser_method_name = ""
        best_fuser_method_mean = None
        candidate_runs: dict[str, list[float]] = {}
        fuser_runs: dict[str, list[float]] = {}
        for record in record_groups.get((dataset, crop_tag), []):
            if record.pipeline_family not in OUR_PIPELINES:
                continue
            candidate_runs.setdefault(record.run_name, []).append(record.test_iou)
            if _is_fusion_only_baseline(record):
                fuser_runs.setdefault(record.run_name, []).append(record.test_iou)
        for run_name, scores in candidate_runs.items():
            candidate_mean = sum(scores) / len(scores)
            if (
                best_single_method_mean is None
                or candidate_mean > best_single_method_mean
            ):
                best_single_method_name = run_name
                best_single_method_mean = candidate_mean
        for run_name, scores in fuser_runs.items():
            candidate_mean = sum(scores) / len(scores)
            if (
                best_fuser_method_mean is None
                or candidate_mean > best_fuser_method_mean
            ):
                best_fuser_method_name = run_name
                best_fuser_method_mean = candidate_mean

        score_rows.append(
            [
                dataset,
                crop_tag,
                _fmt_float(silver_mean),
                _fmt_float(comp_mean),
                _fmt_float(best_fuser_method_mean),
                _fmt_float(best_single_method_mean),
                _fmt_float(oracle_ours_mean),
                _fmt_float(
                    None
                    if best_single_method_mean is None or best_fuser_method_mean is None
                    else best_single_method_mean - best_fuser_method_mean
                ),
                _fmt_float(
                    None
                    if best_single_method_mean is None
                    else best_single_method_mean - silver_mean
                ),
                _fmt_float(
                    None
                    if best_single_method_mean is None
                    else best_single_method_mean - comp_mean
                ),
            ]
        )
        name_rows.append(
            [
                dataset,
                crop_tag,
                best_fuser_method_name,
                best_single_method_name,
            ]
        )

    print("\nPer-crop average scores")
    print(
        tabulate(
            score_rows,
            headers=[
                "dataset",
                "crop",
                "silver_mean",
                "best_comp_mean",
                "fuser_mean",
                "best_single_mean",
                "foldwise_best_mean",
                "gap_fuser",
                "gap_silver",
                "gap_comp",
            ],
            tablefmt="github",
        )
    )
    print("\nPer-crop average methods")
    print(
        tabulate(
            name_rows,
            headers=[
                "dataset",
                "crop",
                "best_fuser",
                "best_ours",
            ],
            tablefmt="github",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize final MLflow leaderboard from an exported SQLite DB."
    )
    parser.add_argument("db_path", type=Path, help="Path to exported MLflow SQLite DB.")
    args = parser.parse_args()

    records = load_final_runs(args.db_path)
    summaries = build_fold_summaries(records)
    print_fold_table(summaries)
    print_crop_average_table(summaries, records)


if __name__ == "__main__":
    main()
