#!/usr/bin/env python3
"""Preflight readiness checks for ablation datasets.

This script intentionally runs before the expensive ablation workflow.  It
checks that DVC-produced preprocessing artifacts exist, that split assignments
are valid at the cell/crop level, and that the configured crop size covers the
GT cell bounding boxes well enough for the run to be meaningful.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tifffile
import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.run_ablation import load_config  # noqa: E402
from silver_truth.data_processing.segmentation_stats import (  # noqa: E402
    collect_segmentation_object_stats,
)
from silver_truth.evaluation.preflight import (  # noqa: E402
    build_split_sanity_audit,
    write_split_sanity_bundle,
)


DEFAULT_CROP_SIZE_GRID = (32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512)


@dataclass(frozen=True)
class FoldArtifacts:
    split_name: str
    whole_image_parquet: Path
    qa_parquet: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a dataset/config is ready for reduced/full ablation "
            "experiments."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment variant YAML, for example experiments/variants/hela_crop256.yaml.",
    )
    parser.add_argument(
        "--fold",
        dest="folds",
        action="append",
        default=None,
        help="Fold to check: 1, 2, fold-1, fold-2, or mixed. Repeatable. Default: 1 and 2.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "synchronized_data",
        help="Root containing synchronized datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "audits" / "dataset_readiness",
        help="Directory where readiness reports are written.",
    )
    parser.add_argument(
        "--min-crop-fit",
        type=float,
        default=0.95,
        help="Minimum acceptable fraction of GT bboxes fitting the configured crop.",
    )
    parser.add_argument(
        "--crop-size",
        dest="extra_crop_sizes",
        type=int,
        action="append",
        default=None,
        help="Additional crop size to include in the crop-fit comparison.",
    )
    parser.add_argument(
        "--max-crop-file-checks",
        type=int,
        default=200,
        help=(
            "Maximum QA crop TIFFs per fold to check for existence and dimensions. "
            "Use 0 to skip or -1 to check all rows."
        ),
    )
    parser.add_argument(
        "--no-require-jaccard",
        action="store_true",
        help="Do not fail when split QA parquets are missing jaccard_score.",
    )
    parser.add_argument(
        "--repro",
        action="store_true",
        help="Run dvc repro for missing known preprocessing stages before checking.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 after writing reports.",
    )
    return parser.parse_args()


def _normalize_split_name(fold: str) -> str:
    normalized = str(fold).strip().lower()
    if normalized in {"1", "fold-1"}:
        return "fold-1"
    if normalized in {"2", "fold-2"}:
        return "fold-2"
    if normalized == "mixed":
        return "mixed"
    raise ValueError(f"Unsupported fold: {fold}")


def _dvc_fold_stage(split_name: str, dataset: str) -> str:
    suffix = {"fold-1": "fold1", "fold-2": "fold2", "mixed": "mixed"}[split_name]
    return f"create_{suffix}@{dataset}"


def _dvc_qa_split_stage(split_name: str, matrix_key: str) -> str:
    suffix = {"fold-1": "fold1", "fold-2": "fold2", "mixed": "mixed"}[split_name]
    return f"create_qa_crops_split_{suffix}@{matrix_key}"


def _load_qa_crop_matrix(dvc_yaml: Path) -> dict[tuple[str, int], str]:
    if not dvc_yaml.exists():
        return {}
    payload = yaml.safe_load(dvc_yaml.read_text(encoding="utf-8")) or {}
    matrix: dict[str, dict[str, Any]] = {}
    for item in payload.get("vars", []):
        if "qa_crop_matrix" in item:
            matrix = item["qa_crop_matrix"] or {}
            break
    return {
        (str(value["dataset"]), int(value["crop_size"])): str(key)
        for key, value in matrix.items()
    }


def _resolve_path(path_value: object, root: Path = PROJECT_ROOT) -> Path | None:
    if path_value is None or pd.isna(path_value):
        return None
    path = Path(str(path_value))
    return path if path.is_absolute() else root / path


def _dataset_structure_status(dataset_root: Path) -> dict[str, Any]:
    required_dirs = [
        dataset_root / "01",
        dataset_root / "02",
        dataset_root / "01_GT" / "SEG",
        dataset_root / "02_GT" / "SEG",
    ]
    dir_status = {str(path): path.is_dir() for path in required_dirs}
    tif_counts = {
        str(path): len(list(path.glob("*.tif"))) if path.is_dir() else 0
        for path in required_dirs
    }
    return {
        "dataset_root": str(dataset_root),
        "exists": dataset_root.is_dir(),
        "required_dirs": dir_status,
        "tif_counts": tif_counts,
        "ok": dataset_root.is_dir()
        and all(dir_status.values())
        and all(count > 0 for count in tif_counts.values()),
    }


def _crop_profile(
    *,
    dataset_root: Path,
    selected_crop_size: int,
    crop_sizes: tuple[int, ...],
    min_crop_fit: float,
) -> dict[str, Any]:
    gt_dirs = [dataset_root / "01_GT" / "SEG", dataset_root / "02_GT" / "SEG"]
    if not all(path.is_dir() for path in gt_dirs):
        return {
            "ok": False,
            "reason": "Missing GT segmentation directories.",
            "gt_dirs": [str(path) for path in gt_dirs],
        }

    stats_df = collect_segmentation_object_stats(gt_dirs)
    if stats_df.empty:
        return {
            "ok": False,
            "reason": "No GT objects found.",
            "gt_dirs": [str(path) for path in gt_dirs],
        }

    bbox_max = stats_df["bbox_max_dim_px"]
    crop_fit: dict[str, Any] = {}
    for crop_size in crop_sizes:
        fits = (stats_df["bbox_height_px"] <= crop_size) & (
            stats_df["bbox_width_px"] <= crop_size
        )
        crop_fit[str(crop_size)] = {
            "fit_rate": float(fits.mean()),
            "over_count": int((~fits).sum()),
        }

    selected = crop_fit[str(selected_crop_size)]
    return {
        "ok": bool(selected["fit_rate"] >= min_crop_fit),
        "min_crop_fit": float(min_crop_fit),
        "selected_crop_size": int(selected_crop_size),
        "selected_crop_fit_rate": selected["fit_rate"],
        "selected_crop_over_count": selected["over_count"],
        "n_gt_cells": int(len(stats_df)),
        "n_gt_frames": int(stats_df["file_path"].nunique()),
        "p90_bbox_max_dim_px": int(bbox_max.quantile(0.90).round()),
        "p95_bbox_max_dim_px": int(bbox_max.quantile(0.95).round()),
        "p99_bbox_max_dim_px": int(bbox_max.quantile(0.99).round()),
        "max_bbox_dim_px": int(bbox_max.max()),
        "exact_crop_for_100pct": int(bbox_max.max()),
        "crop_fit": crop_fit,
    }


def _parquet_schema_status(
    path: Path,
    *,
    required_columns: set[str],
    recommended_columns: set[str] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "ok": False}
    df = pd.read_parquet(path)
    missing = sorted(required_columns - set(df.columns))
    missing_recommended = sorted((recommended_columns or set()) - set(df.columns))
    split_counts = (
        df["split"].value_counts(dropna=False).to_dict()
        if "split" in df.columns
        else {}
    )
    return {
        "path": str(path),
        "exists": True,
        "ok": not missing,
        "n_rows": int(len(df)),
        "columns": sorted(df.columns.tolist()),
        "missing_required_columns": missing,
        "missing_recommended_columns": missing_recommended,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
    }


def _sample_crop_file_status(
    qa_parquet: Path,
    *,
    crop_size: int,
    max_checks: int,
) -> dict[str, Any]:
    if max_checks == 0:
        return {"skipped": True, "ok": True}
    if not qa_parquet.exists():
        return {"skipped": False, "ok": False, "reason": "QA parquet is missing."}

    df = pd.read_parquet(qa_parquet)
    if "stacked_path" not in df.columns:
        return {
            "skipped": False,
            "ok": False,
            "reason": "QA parquet has no stacked_path column.",
        }

    rows = df[["stacked_path"]].dropna()
    if max_checks > 0:
        rows = rows.head(max_checks)

    missing: list[str] = []
    bad_shapes: list[dict[str, Any]] = []
    checked = 0
    for row in rows.itertuples(index=False):
        crop_path = _resolve_path(row.stacked_path)
        if crop_path is None or not crop_path.exists():
            missing.append(str(row.stacked_path))
            continue
        checked += 1
        image = tifffile.imread(crop_path)
        if tuple(image.shape[-2:]) != (crop_size, crop_size):
            bad_shapes.append(
                {
                    "path": str(crop_path),
                    "shape": [int(value) for value in image.shape],
                }
            )

    return {
        "skipped": False,
        "ok": not missing and not bad_shapes,
        "checked": checked,
        "sample_size": int(len(rows)),
        "missing_count": len(missing),
        "missing_samples": missing[:10],
        "bad_shape_count": len(bad_shapes),
        "bad_shape_samples": bad_shapes[:10],
    }


def _build_fold_artifacts(
    config_path: Path, folds: list[str]
) -> tuple[dict[str, Any], list[FoldArtifacts]]:
    first_cfg = load_config(config_path, folds[0])
    artifacts: list[FoldArtifacts] = []
    for fold in folds:
        cfg = load_config(config_path, fold)
        artifacts.append(
            FoldArtifacts(
                split_name=cfg["split_name"],
                whole_image_parquet=Path(cfg["whole_image_parquet_template"]),
                qa_parquet=Path(cfg["qa_parquet_template"]),
            )
        )
    return first_cfg, artifacts


def _missing_artifact_stages(
    *,
    dataset: str,
    crop_size: int,
    fold_artifacts: list[FoldArtifacts],
    matrix_key: str | None,
) -> list[str]:
    stages: list[str] = []
    for artifact in fold_artifacts:
        if not artifact.whole_image_parquet.exists():
            stages.append(_dvc_fold_stage(artifact.split_name, dataset))
        if not artifact.qa_parquet.exists() and matrix_key is not None:
            stages.append(f"create_qa_crops_base@{matrix_key}")
            stages.append(_dvc_qa_split_stage(artifact.split_name, matrix_key))
    return sorted(set(stages))


def _run_dvc_repro(stages: list[str]) -> None:
    if not stages:
        return
    subprocess.run(["dvc", "repro", *stages], cwd=PROJECT_ROOT, check=True)


def _markdown_report(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        f"# Dataset Readiness - {cfg['dataset']} sz{cfg['crop_size']}",
        "",
        f"- Config: `{cfg['config_path']}`",
        f"- Overall ready: `{report['ready']}`",
        f"- Dataset structure: `{'PASS' if report['dataset_structure']['ok'] else 'FAIL'}`",
        f"- Crop fit gate: `{'PASS' if report['crop_profile'].get('ok') else 'FAIL'}`",
        "",
    ]

    if report["missing_dvc_stages"]:
        lines.extend(
            [
                "## Missing DVC Stages",
                "",
                "Run:",
                "",
                "```bash",
                "dvc repro " + " ".join(report["missing_dvc_stages"]),
                "```",
                "",
            ]
        )

    if report.get("dataset_pull_command"):
        lines.extend(
            [
                "## Missing Dataset",
                "",
                "Run:",
                "",
                "```bash",
                report["dataset_pull_command"],
                "```",
                "",
            ]
        )

    crop = report["crop_profile"]
    lines.extend(["## Crop Profile", ""])
    if crop.get("ok") is not None and "selected_crop_fit_rate" in crop:
        lines.extend(
            [
                f"- Selected crop: `{crop['selected_crop_size']}`",
                f"- Fit rate: `{crop['selected_crop_fit_rate']:.4f}`",
                f"- Oversized GT cells: `{crop['selected_crop_over_count']}`",
                f"- P95 bbox max dim: `{crop['p95_bbox_max_dim_px']}`",
                f"- P99 bbox max dim: `{crop['p99_bbox_max_dim_px']}`",
                f"- Exact crop for 100% bbox fit: `{crop['exact_crop_for_100pct']}`",
                "",
            ]
        )
    else:
        lines.extend([f"- {crop.get('reason', 'Crop profile unavailable.')}", ""])

    lines.extend(["## Fold Gates", ""])
    lines.append("| Fold | Whole parquet | QA parquet | Split sanity | Crop files |")
    lines.append("|---|---|---|---|---|")
    for fold in report["folds"]:
        lines.append(
            "| {split} | {whole} | {qa} | {split_ok} | {crop_ok} |".format(
                split=fold["split_name"],
                whole="PASS" if fold["whole_image_status"]["ok"] else "FAIL",
                qa="PASS" if fold["qa_status"]["ok"] else "FAIL",
                split_ok="PASS"
                if fold["split_sanity"].get("scientifically_valid")
                else "FAIL",
                crop_ok="PASS" if fold["crop_file_status"]["ok"] else "FAIL",
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    config_path = (
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    )
    folds = [_normalize_split_name(fold) for fold in (args.folds or ["1", "2"])]

    cfg, fold_artifacts = _build_fold_artifacts(config_path, folds)
    dataset = str(cfg["dataset"])
    crop_size = int(cfg["crop_size"])
    dataset_root = args.datasets_root / dataset

    qa_crop_matrix = _load_qa_crop_matrix(PROJECT_ROOT / "dvc.yaml")
    matrix_key = qa_crop_matrix.get((dataset, crop_size))
    missing_stages = _missing_artifact_stages(
        dataset=dataset,
        crop_size=crop_size,
        fold_artifacts=fold_artifacts,
        matrix_key=matrix_key,
    )

    if args.repro and missing_stages:
        _run_dvc_repro(missing_stages)
        missing_stages = _missing_artifact_stages(
            dataset=dataset,
            crop_size=crop_size,
            fold_artifacts=fold_artifacts,
            matrix_key=matrix_key,
        )

    output_dir = args.output_dir / dataset / f"sz{crop_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_sizes = tuple(
        sorted(
            set(
                DEFAULT_CROP_SIZE_GRID
                + (crop_size,)
                + tuple(args.extra_crop_sizes or [])
            )
        )
    )
    crop_profile = _crop_profile(
        dataset_root=dataset_root,
        selected_crop_size=crop_size,
        crop_sizes=crop_sizes,
        min_crop_fit=args.min_crop_fit,
    )

    fold_reports: list[dict[str, Any]] = []
    for artifact in fold_artifacts:
        whole_status = _parquet_schema_status(
            artifact.whole_image_parquet,
            required_columns={"split", "has_gt", "gt_cell_count", "gt_image"},
            recommended_columns={"source_image", "campaign_number", "time_frame"},
        )
        qa_required = {"split", "gt_image", "label", "stacked_path"}
        if not args.no_require_jaccard:
            qa_required.add("jaccard_score")
        qa_status = _parquet_schema_status(
            artifact.qa_parquet,
            required_columns=qa_required,
            recommended_columns={"cell_id", "campaign_number", "time_frame"},
        )
        crop_file_status = _sample_crop_file_status(
            artifact.qa_parquet,
            crop_size=crop_size,
            max_checks=args.max_crop_file_checks,
        )

        split_sanity: dict[str, Any]
        if whole_status["ok"] and qa_status["ok"]:
            split_sanity, sample_manifest = build_split_sanity_audit(
                dataset=dataset,
                crop_size=crop_size,
                split_name=artifact.split_name,
                whole_image_parquet=artifact.whole_image_parquet,
                qa_parquet=artifact.qa_parquet,
            )
            write_split_sanity_bundle(
                split_sanity,
                sample_manifest,
                output_dir / artifact.split_name / "split_sanity",
            )
        else:
            split_sanity = {
                "scientifically_valid": False,
                "reason": "Skipped because required parquets/columns are missing.",
            }

        fold_reports.append(
            {
                "split_name": artifact.split_name,
                "whole_image_status": whole_status,
                "qa_status": qa_status,
                "crop_file_status": crop_file_status,
                "split_sanity": split_sanity,
            }
        )

    dataset_structure = _dataset_structure_status(dataset_root)
    dataset_dvc_file = PROJECT_ROOT / "data" / "synchronized_data" / f"{dataset}.dvc"
    dataset_pull_command = (
        f"dvc pull data/synchronized_data/{dataset}.dvc"
        if not dataset_structure["ok"] and dataset_dvc_file.exists()
        else None
    )
    ready = (
        dataset_structure["ok"]
        and not missing_stages
        and bool(crop_profile.get("ok"))
        and all(fold["whole_image_status"]["ok"] for fold in fold_reports)
        and all(fold["qa_status"]["ok"] for fold in fold_reports)
        and all(fold["crop_file_status"]["ok"] for fold in fold_reports)
        and all(
            fold["split_sanity"].get("scientifically_valid", False)
            for fold in fold_reports
        )
    )

    report = {
        "ready": bool(ready),
        "config": {
            "config_path": str(config_path),
            "dataset": dataset,
            "crop_size": crop_size,
            "matrix_key": matrix_key,
        },
        "dataset_structure": dataset_structure,
        "dataset_pull_command": dataset_pull_command,
        "missing_dvc_stages": missing_stages,
        "crop_profile": crop_profile,
        "folds": fold_reports,
    }

    json_path = output_dir / "readiness_report.json"
    md_path = output_dir / "readiness_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    print(json.dumps({"ready": ready, "report": str(json_path)}, indent=2))
    print(f"Wrote Markdown report: {md_path}")
    if missing_stages:
        print("Missing preprocessing stages:")
        print("  dvc repro " + " ".join(missing_stages))

    if ready or args.no_fail:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
