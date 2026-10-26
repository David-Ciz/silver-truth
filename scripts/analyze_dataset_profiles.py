from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import tifffile

from silver_truth.data_processing.segmentation_stats import (
    collect_segmentation_object_stats,
    summarize_segmentation_object_stats,
)
from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    process_dataset_directory,
)


DEFAULT_DATASETS = (
    "DIC-C2DH-HeLa",
    "BF-C2DL-HSC",
    "BF-C2DL-MuSC",
)
DEFAULT_CROP_SIZES = (32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile synchronized datasets and compare GT cell sizes against a "
            "range of crop sizes."
        )
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("data/synchronized_data"),
        help="Root containing synchronized datasets.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_names",
        action="append",
        help=(
            "Dataset name under --datasets-root. Can be passed multiple times. "
            "Defaults to HeLa, HSC, and MuSC."
        ),
    )
    parser.add_argument(
        "--crop-size",
        dest="crop_sizes",
        action="append",
        type=int,
        help=(
            "Square crop size to evaluate. Can be passed multiple times. "
            "Defaults to a broad comparison grid."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/dataset_profiles"),
        help="Directory where reports and tables will be written.",
    )
    return parser.parse_args()


def _find_sequence_dirs(dataset_root: Path) -> list[Path]:
    return [dataset_root / "01", dataset_root / "02"]


def _find_gt_dirs(dataset_root: Path) -> list[Path]:
    return [dataset_root / "01_GT" / "SEG", dataset_root / "02_GT" / "SEG"]


def _read_image_shape_and_dtype(image_path: Path) -> tuple[tuple[int, ...], str]:
    image = tifffile.imread(image_path)
    return tuple(int(v) for v in image.shape), str(image.dtype)


def _sequence_profile(sequence_dir: Path) -> dict[str, object]:
    tif_paths = sorted(sequence_dir.glob("*.tif"))
    shape_counter: Counter[str] = Counter()
    dtype_counter: Counter[str] = Counter()

    for tif_path in tif_paths:
        shape, dtype = _read_image_shape_and_dtype(tif_path)
        shape_counter[str(shape)] += 1
        dtype_counter[dtype] += 1

    return {
        "sequence": sequence_dir.name,
        "n_frames": len(tif_paths),
        "shape_counts": dict(shape_counter),
        "dtype_counts": dict(dtype_counter),
    }


def _choose_uniform_shape(sequence_profiles: list[dict[str, object]]) -> str | None:
    shape_keys: set[str] = set()
    for profile in sequence_profiles:
        shape_keys.update(profile["shape_counts"].keys())
    if len(shape_keys) == 1:
        return next(iter(shape_keys))
    return None


def _derive_crop_thresholds(stats_df: pd.DataFrame, crop_sizes: tuple[int, ...]) -> dict[str, object]:
    bbox_max_dim = stats_df["bbox_max_dim_px"]
    quantiles = {
        "p90_bbox_max_dim_px": int(math.ceil(float(bbox_max_dim.quantile(0.90)))),
        "p95_bbox_max_dim_px": int(math.ceil(float(bbox_max_dim.quantile(0.95)))),
        "p99_bbox_max_dim_px": int(math.ceil(float(bbox_max_dim.quantile(0.99)))),
        "max_bbox_dim_px": int(bbox_max_dim.max()),
    }

    thresholds: dict[str, object] = {}
    for target, label in ((0.95, "min_tested_crop_for_95pct"), (0.99, "min_tested_crop_for_99pct"), (1.0, "min_tested_crop_for_100pct")):
        chosen = None
        for crop_size in crop_sizes:
            fit_rate = float(
                (
                    (stats_df["bbox_height_px"] <= crop_size)
                    & (stats_df["bbox_width_px"] <= crop_size)
                ).mean()
            )
            if fit_rate >= target:
                chosen = crop_size
                break
        thresholds[label] = chosen

    thresholds["exact_crop_for_100pct"] = int(bbox_max_dim.max())
    thresholds.update(quantiles)
    return thresholds


def _build_dataset_profile(dataset_root: Path, crop_sizes: tuple[int, ...]) -> tuple[dict[str, object], pd.DataFrame]:
    dataset_name = dataset_root.name
    dataset_info, competitors = process_dataset_directory(dataset_root)

    sequence_dirs = _find_sequence_dirs(dataset_root)
    gt_dirs = _find_gt_dirs(dataset_root)
    sequence_profiles = [_sequence_profile(sequence_dir) for sequence_dir in sequence_dirs]

    stats_df = collect_segmentation_object_stats(gt_dirs).copy()
    stats_df.insert(0, "dataset", dataset_name)

    crop_summary = summarize_segmentation_object_stats(stats_df, crop_sizes=crop_sizes)
    overall = crop_summary["overall"]

    cells_per_frame = (
        stats_df.groupby("file_path")["label_id"].count().rename("cells_per_frame")
    )
    thresholds = _derive_crop_thresholds(stats_df, crop_sizes)
    n_raw_frames = sum(int(profile["n_frames"]) for profile in sequence_profiles)

    profile = {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "n_rows_in_dataset_table": len(dataset_info),
        "n_raw_frames": n_raw_frames,
        "n_competitors": len(competitors),
        "competitors": sorted(competitors),
        "has_silver_truth": (dataset_root / "01_ST" / "SEG").is_dir()
        or (dataset_root / "02_ST" / "SEG").is_dir(),
        "sequence_profiles": sequence_profiles,
        "uniform_image_shape": _choose_uniform_shape(sequence_profiles),
        "n_gt_frames": int(overall["n_files"]),
        "n_gt_cells": int(overall["n_cells"]),
        "annotated_frame_fraction": float(overall["n_files"] / n_raw_frames) if n_raw_frames else 0.0,
        "mean_cells_per_frame": float(cells_per_frame.mean()),
        "median_cells_per_frame": float(cells_per_frame.median()),
        "p95_cells_per_frame": float(cells_per_frame.quantile(0.95)),
        "mean_area_px": float(overall["mean_area_px"]),
        "median_area_px": float(overall["median_area_px"]),
        "p95_area_px": float(overall["p95_area_px"]),
        "mean_bbox_height_px": float(overall["mean_bbox_height_px"]),
        "mean_bbox_width_px": float(overall["mean_bbox_width_px"]),
        "p95_bbox_height_px": float(overall["p95_bbox_height_px"]),
        "p95_bbox_width_px": float(overall["p95_bbox_width_px"]),
        "mean_bbox_aspect_ratio_wh": float(overall["mean_bbox_aspect_ratio_wh"]),
        "p95_bbox_aspect_ratio_wh": float(overall["p95_bbox_aspect_ratio_wh"]),
        "mean_bbox_elongation_ratio": float(overall["mean_bbox_elongation_ratio"]),
        "p95_bbox_elongation_ratio": float(overall["p95_bbox_elongation_ratio"]),
        "wide_bbox_rate": float(overall["wide_bbox_rate"]),
        "tall_bbox_rate": float(overall["tall_bbox_rate"]),
        "square_bbox_rate": float(overall["square_bbox_rate"]),
        "crop_fit": {
            str(crop_size): {
                "fit_rate": float(overall[f"fit_bbox_rate_sz{crop_size}"]),
                "over_count": int(overall[f"over_bbox_count_sz{crop_size}"]),
            }
            for crop_size in crop_sizes
        },
        "crop_thresholds": thresholds,
    }
    return profile, stats_df


def _build_comparison_tables(
    profiles: list[dict[str, object]],
    crop_sizes: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    crop_rows: list[dict[str, object]] = []

    for profile in profiles:
        thresholds = profile["crop_thresholds"]
        summary_rows.append(
            {
                "dataset": profile["dataset"],
                "uniform_image_shape": profile["uniform_image_shape"],
                "n_raw_frames": profile["n_raw_frames"],
                "n_gt_frames": profile["n_gt_frames"],
                "annotated_frame_fraction": profile["annotated_frame_fraction"],
                "n_gt_cells": profile["n_gt_cells"],
                "n_competitors": profile["n_competitors"],
                "has_silver_truth": profile["has_silver_truth"],
                "mean_cells_per_frame": profile["mean_cells_per_frame"],
                "median_cells_per_frame": profile["median_cells_per_frame"],
                "p95_cells_per_frame": profile["p95_cells_per_frame"],
                "mean_area_px": profile["mean_area_px"],
                "median_area_px": profile["median_area_px"],
                "p95_area_px": profile["p95_area_px"],
                "mean_bbox_height_px": profile["mean_bbox_height_px"],
                "mean_bbox_width_px": profile["mean_bbox_width_px"],
                "p95_bbox_height_px": profile["p95_bbox_height_px"],
                "p95_bbox_width_px": profile["p95_bbox_width_px"],
                "p90_bbox_max_dim_px": thresholds["p90_bbox_max_dim_px"],
                "p95_bbox_max_dim_px": thresholds["p95_bbox_max_dim_px"],
                "p99_bbox_max_dim_px": thresholds["p99_bbox_max_dim_px"],
                "max_bbox_dim_px": thresholds["max_bbox_dim_px"],
                "min_tested_crop_for_95pct": thresholds["min_tested_crop_for_95pct"],
                "min_tested_crop_for_99pct": thresholds["min_tested_crop_for_99pct"],
                "min_tested_crop_for_100pct": thresholds["min_tested_crop_for_100pct"],
                "exact_crop_for_100pct": thresholds["exact_crop_for_100pct"],
                "wide_bbox_rate": profile["wide_bbox_rate"],
                "tall_bbox_rate": profile["tall_bbox_rate"],
                "square_bbox_rate": profile["square_bbox_rate"],
            }
        )

        for crop_size in crop_sizes:
            crop_fit = profile["crop_fit"][str(crop_size)]
            crop_rows.append(
                {
                    "dataset": profile["dataset"],
                    "crop_size": crop_size,
                    "fit_rate": crop_fit["fit_rate"],
                    "over_count": crop_fit["over_count"],
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(crop_rows)


def _markdown_report(
    profiles: list[dict[str, object]],
    comparison_df: pd.DataFrame,
    crop_df: pd.DataFrame,
) -> str:
    lines = [
        "# Dataset Profile Comparison",
        "",
        "Datasets: " + ", ".join(profile["dataset"] for profile in profiles),
        "",
        "## Summary Table",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## Crop Fit Table",
        "",
        crop_df.to_markdown(index=False),
        "",
    ]

    for profile in profiles:
        lines.extend(
            [
                f"## {profile['dataset']}",
                "",
                f"- Dataset root: `{profile['dataset_root']}`",
                f"- Competitors ({profile['n_competitors']}): {', '.join(profile['competitors'])}",
                f"- Uniform image shape: `{profile['uniform_image_shape']}`",
                f"- Raw frames: {profile['n_raw_frames']}",
                f"- GT frames: {profile['n_gt_frames']}",
                f"- Annotation coverage: {profile['annotated_frame_fraction']:.3%}",
                f"- GT cells: {profile['n_gt_cells']}",
                f"- Mean cells/frame: {profile['mean_cells_per_frame']:.2f}",
                f"- P95 bbox max dim: {profile['crop_thresholds']['p95_bbox_max_dim_px']} px",
                f"- P99 bbox max dim: {profile['crop_thresholds']['p99_bbox_max_dim_px']} px",
                f"- Exact crop for 100% bbox fit: {profile['crop_thresholds']['exact_crop_for_100pct']} px",
                "",
                "Sequence profiles:",
            ]
        )
        for seq_profile in profile["sequence_profiles"]:
            lines.append(
                f"- {seq_profile['sequence']}: {seq_profile['n_frames']} frames, "
                f"shapes={seq_profile['shape_counts']}, dtypes={seq_profile['dtype_counts']}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    dataset_names = tuple(args.dataset_names or DEFAULT_DATASETS)
    crop_sizes = tuple(sorted(set(args.crop_sizes or DEFAULT_CROP_SIZES)))
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles: list[dict[str, object]] = []
    stats_frames: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        dataset_root = args.datasets_root / dataset_name
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
        profile, stats_df = _build_dataset_profile(dataset_root, crop_sizes)
        profiles.append(profile)
        stats_frames.append(stats_df)

    comparison_df, crop_df = _build_comparison_tables(profiles, crop_sizes)
    all_stats_df = pd.concat(stats_frames, ignore_index=True)

    summary_path = output_dir / "dataset_profiles_summary.json"
    comparison_path = output_dir / "dataset_comparison.csv"
    crop_path = output_dir / "crop_fit_comparison.csv"
    per_object_path = output_dir / "per_object_stats.parquet"
    report_path = output_dir / "report.md"

    summary_payload = {
        "datasets": profiles,
        "crop_sizes": list(crop_sizes),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    comparison_df.to_csv(comparison_path, index=False)
    crop_df.to_csv(crop_path, index=False)
    all_stats_df.to_parquet(per_object_path, index=False)
    report_path.write_text(
        _markdown_report(profiles, comparison_df, crop_df),
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, indent=2))
    print(f"\nSaved summary JSON to {summary_path}")
    print(f"Saved comparison CSV to {comparison_path}")
    print(f"Saved crop-fit CSV to {crop_path}")
    print(f"Saved per-object stats to {per_object_path}")
    print(f"Saved Markdown report to {report_path}")


if __name__ == "__main__":
    main()
