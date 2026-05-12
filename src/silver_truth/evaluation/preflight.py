from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)

_EXPECTED_SEQUENCES = {
    "fold-1": {"train_validation": "01", "test": "02"},
    "fold-2": {"train_validation": "02", "test": "01"},
}


def _normalize_split_name(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    if normalized in {"1", "fold-1"}:
        return "fold-1"
    if normalized in {"2", "fold-2"}:
        return "fold-2"
    return normalized


def _ensure_whole_image_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "has_gt" not in df.columns:
        df["has_gt"] = df.get("gt_image", pd.Series(index=df.index)).notna()
    else:
        df["has_gt"] = df["has_gt"].fillna(False).astype(bool)

    if "gt_cell_count" not in df.columns:
        df["gt_cell_count"] = 0
    else:
        df["gt_cell_count"] = (
            pd.to_numeric(df["gt_cell_count"], errors="coerce").fillna(0).astype(int)
        )

    return df


def _whole_image_split_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split_name in ["train", "validation", "test"]:
        split_df = df[df["split"] == split_name].copy()
        gt_df = split_df[split_df["has_gt"]].copy()
        summary[split_name] = {
            "raw_frames": int(len(split_df)),
            "gt_labeled_frames": int(len(gt_df)),
            "gt_cells": int(gt_df["gt_cell_count"].sum()) if not gt_df.empty else 0,
            "campaigns": sorted(
                split_df["campaign_number"].dropna().astype(str).unique().tolist()
            )
            if "campaign_number" in split_df.columns
            else [],
            "time_frame_min": int(split_df["time_frame"].min())
            if not split_df.empty and "time_frame" in split_df.columns
            else None,
            "time_frame_max": int(split_df["time_frame"].max())
            if not split_df.empty and "time_frame" in split_df.columns
            else None,
            "gt_time_frames": sorted(gt_df["time_frame"].dropna().astype(int).tolist())
            if not gt_df.empty and "time_frame" in gt_df.columns
            else [],
        }
    return summary


def _qa_split_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    real_cell_cols = [
        col for col in ["gt_image", "label", "campaign_number"] if col in df.columns
    ]
    for split_name in ["train", "validation", "test"]:
        split_df = df[df["split"] == split_name].copy()
        summary[split_name] = {
            "qa_candidate_rows": int(len(split_df)),
            "real_cells": int(split_df[real_cell_cols].drop_duplicates().shape[0])
            if real_cell_cols
            else None,
        }
    return summary


def _collect_overlap(values_by_split: dict[str, set[str]]) -> dict[str, list[str]]:
    overlaps: dict[str, list[str]] = {}
    split_names = list(values_by_split.keys())
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            shared = sorted(values_by_split[left] & values_by_split[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared[:10]
    return overlaps


def _gt_split_key(value: Any) -> str:
    """Return a split-alignment key that preserves campaign/path context."""
    return str(value).replace("\\", "/")


def _gt_name_split_mapping(df: pd.DataFrame) -> dict[str, str]:
    subset = df[df["gt_image"].notna()].copy()
    if subset.empty:
        return {}
    subset["gt_key"] = subset["gt_image"].map(_gt_split_key)
    mapping = subset[["gt_key", "split"]].drop_duplicates().set_index("gt_key")["split"]
    return {str(key): str(split) for key, split in mapping.items()}


def _qa_split_alignment_samples(
    whole_df: pd.DataFrame, qa_df: pd.DataFrame, limit: int = 10
) -> list[dict[str, str]]:
    whole_mapping = _gt_name_split_mapping(whole_df)
    qa_subset = qa_df[qa_df["gt_image"].notna()].copy()
    if qa_subset.empty:
        return []

    qa_subset["gt_key"] = qa_subset["gt_image"].map(_gt_split_key)
    qa_mapping = (
        qa_subset[["gt_key", "split"]].drop_duplicates().set_index("gt_key")["split"]
    )

    mismatches: list[dict[str, str]] = []
    for gt_key, qa_split in qa_mapping.items():
        whole_split = whole_mapping.get(str(gt_key))
        if whole_split is None or str(qa_split) != whole_split:
            mismatches.append(
                {
                    "gt_key": str(gt_key),
                    "gt_name": Path(str(gt_key)).name,
                    "qa_split": str(qa_split),
                    "whole_split": str(whole_split)
                    if whole_split is not None
                    else "missing",
                }
            )
        if len(mismatches) >= limit:
            break
    return mismatches


def _sample_manifest(qa_df: pd.DataFrame, max_per_split: int = 8) -> pd.DataFrame:
    columns = [
        col
        for col in [
            "split",
            "cell_id",
            "campaign_number",
            "original_image_key",
            "label",
            "gt_image",
            "stacked_path",
            "time_frame",
        ]
        if col in qa_df.columns
    ]
    sampled = (
        qa_df.sort_values(
            [
                col
                for col in ["split", "campaign_number", "time_frame", "cell_id"]
                if col in qa_df.columns
            ]
        )
        .groupby("split", group_keys=False)
        .head(max_per_split)
    )
    return sampled[columns].copy() if columns else pd.DataFrame()


def build_split_sanity_audit(
    *,
    dataset: str,
    crop_size: int,
    split_name: str,
    whole_image_parquet: Path,
    qa_parquet: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    normalized_split = _normalize_split_name(split_name)
    whole_df = _ensure_whole_image_columns(
        load_dataframe_from_parquet_with_metadata(str(whole_image_parquet))
    )
    qa_df = pd.read_parquet(qa_parquet)

    if "split" not in whole_df.columns or "split" not in qa_df.columns:
        raise ValueError(
            "Both whole-image and QA parquets must contain a 'split' column."
        )

    whole_summary = _whole_image_split_summary(whole_df)
    qa_summary = _qa_split_summary(qa_df)

    held_in_gt_cells = (
        whole_summary["train"]["gt_cells"] + whole_summary["validation"]["gt_cells"]
    )
    held_in_qa_rows = (
        qa_summary["train"]["qa_candidate_rows"]
        + qa_summary["validation"]["qa_candidate_rows"]
    )
    val_gt_fraction = whole_summary["validation"]["gt_cells"] / max(held_in_gt_cells, 1)
    val_qa_fraction = qa_summary["validation"]["qa_candidate_rows"] / max(
        held_in_qa_rows, 1
    )

    expected = _EXPECTED_SEQUENCES.get(normalized_split, {})
    train_campaigns = set(whole_summary["train"]["campaigns"])
    validation_campaigns = set(whole_summary["validation"]["campaigns"])
    test_campaigns = set(whole_summary["test"]["campaigns"])

    gt_overlap = _collect_overlap(
        {
            split: set(
                whole_df.loc[
                    (whole_df["split"] == split) & whole_df["gt_image"].notna(),
                    "gt_image",
                ]
                .astype(str)
                .tolist()
            )
            for split in ["train", "validation", "test"]
        }
    )
    cell_key_column = "cell_id" if "cell_id" in qa_df.columns else None
    cell_overlap = (
        _collect_overlap(
            {
                split: set(
                    qa_df.loc[qa_df["split"] == split, cell_key_column]
                    .astype(str)
                    .tolist()
                )
                for split in ["train", "validation", "test"]
            }
        )
        if cell_key_column is not None
        else {}
    )
    qa_alignment_mismatches = _qa_split_alignment_samples(whole_df, qa_df)

    hard_checks = {
        "qa_split_matches_whole_image_split": not qa_alignment_mismatches,
        "expected_train_validation_sequence": not expected
        or (train_campaigns | validation_campaigns) == {expected["train_validation"]},
        "expected_test_sequence": not expected or test_campaigns == {expected["test"]},
        "test_sequence_overlaps_train_validation": (
            train_campaigns | validation_campaigns
        ).isdisjoint(test_campaigns),
        "train_gt_cells_gt_validation": whole_summary["train"]["gt_cells"]
        > whole_summary["validation"]["gt_cells"],
        "train_qa_rows_gt_validation": qa_summary["train"]["qa_candidate_rows"]
        > qa_summary["validation"]["qa_candidate_rows"],
        "validation_gt_fraction_min": val_gt_fraction >= 0.10,
        "validation_gt_fraction_max": val_gt_fraction <= 0.40,
        "validation_qa_fraction_min": val_qa_fraction >= 0.10,
        "validation_qa_fraction_max": val_qa_fraction <= 0.40,
        "train_gt_labeled_frames_min": whole_summary["train"]["gt_labeled_frames"] >= 2,
        "validation_gt_labeled_frames_min": whole_summary["validation"][
            "gt_labeled_frames"
        ]
        >= 1,
        "test_gt_labeled_frames_min": whole_summary["test"]["gt_labeled_frames"] >= 1,
        "no_gt_overlap": not gt_overlap,
        "no_cell_overlap": not cell_overlap,
    }

    audit = {
        "dataset": dataset,
        "crop_size": int(crop_size),
        "split_name": normalized_split,
        "whole_image_parquet": str(whole_image_parquet),
        "qa_parquet": str(qa_parquet),
        "split_strategy": whole_df.attrs.get("split_strategy"),
        "split_unit": whole_df.attrs.get("split_unit"),
        "target_val_fraction": whole_df.attrs.get("target_val_fraction"),
        "actual_val_fraction_by_gt_cells": float(val_gt_fraction),
        "actual_val_fraction_by_qa_rows": float(val_qa_fraction),
        "whole_image_summary": whole_summary,
        "qa_summary": qa_summary,
        "gt_overlap_samples": gt_overlap,
        "cell_overlap_samples": cell_overlap,
        "qa_gt_split_mismatch_samples": qa_alignment_mismatches,
        "hard_checks": hard_checks,
        "scientifically_valid": all(hard_checks.values()),
    }
    return audit, _sample_manifest(qa_df)


def _audit_markdown(audit: dict[str, Any]) -> str:
    whole = audit["whole_image_summary"]
    qa = audit["qa_summary"]
    lines = [
        f"# Split Sanity Audit — {audit['dataset']} {audit['split_name']}",
        "",
        f"- Crop size: {audit['crop_size']}",
        f"- Scientifically valid: {audit['scientifically_valid']}",
        f"- Actual validation fraction by GT cells: {audit['actual_val_fraction_by_gt_cells']:.4f}",
        f"- Actual validation fraction by QA rows: {audit['actual_val_fraction_by_qa_rows']:.4f}",
        "",
        "| Unit | Train | Validation | Test |",
        "|---|---:|---:|---:|",
        f"| raw frames | {whole['train']['raw_frames']} | {whole['validation']['raw_frames']} | {whole['test']['raw_frames']} |",
        f"| GT-labeled frames | {whole['train']['gt_labeled_frames']} | {whole['validation']['gt_labeled_frames']} | {whole['test']['gt_labeled_frames']} |",
        f"| GT cells | {whole['train']['gt_cells']} | {whole['validation']['gt_cells']} | {whole['test']['gt_cells']} |",
        f"| QA candidate rows | {qa['train']['qa_candidate_rows']} | {qa['validation']['qa_candidate_rows']} | {qa['test']['qa_candidate_rows']} |",
        f"| real cells | {qa['train']['real_cells']} | {qa['validation']['real_cells']} | {qa['test']['real_cells']} |",
        "",
        "## Hard checks",
        "",
    ]
    for name, passed in audit["hard_checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines) + "\n"


def write_split_sanity_bundle(
    audit: dict[str, Any], sample_manifest: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "split_sanity.json"
    md_path = output_dir / "split_sanity.md"
    csv_path = output_dir / "sample_manifest.csv"

    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_audit_markdown(audit), encoding="utf-8")
    sample_manifest.to_csv(csv_path, index=False)

    return {
        "json": json_path,
        "markdown": md_path,
        "sample_manifest": csv_path,
    }
