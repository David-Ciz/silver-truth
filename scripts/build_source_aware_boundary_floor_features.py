#!/usr/bin/env python3
"""Add source-aware object-vote boundary floor columns to feature parquets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-parquet", action="append", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument(
        "--feature-pattern",
        default="**/*_paper_ready_pairwise_ranked.parquet",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-column", default="candidate_source")
    parser.add_argument("--target-column", default="true_iou")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--calibration-split", action="append", default=None)
    parser.add_argument("--floor-column", default="object_vote_boundary_floor")
    parser.add_argument("--low-quality-floor", type=float, default=0.25)
    parser.add_argument("--high-quality-floor", type=float, default=0.90)
    parser.add_argument("--quality-quantile", type=float, default=0.50)
    parser.add_argument(
        "--invert",
        action="store_true",
        help=(
            "Assign high floor to low-quality sources and low floor to "
            "high-quality sources. Default is low-quality -> low floor."
        ),
    )
    args = parser.parse_args()

    inputs = list(args.feature_parquet or [])
    if args.feature_root is not None:
        inputs.extend(sorted(args.feature_root.glob(args.feature_pattern)))
    if not inputs:
        raise SystemExit("No feature parquets were provided or discovered.")

    paths = []
    relative_root = args.feature_root if args.feature_root is not None else None
    for path in inputs:
        out = add_source_aware_floor(
            path,
            output_root=args.output_root,
            relative_root=relative_root,
            source_column=args.source_column,
            target_column=args.target_column,
            split_column=args.split_column,
            calibration_splits=tuple(args.calibration_split or ("train", "validation")),
            floor_column=args.floor_column,
            low_quality_floor=args.low_quality_floor,
            high_quality_floor=args.high_quality_floor,
            quality_quantile=args.quality_quantile,
            invert=args.invert,
        )
        paths.append(out)
        print(out)


def add_source_aware_floor(
    path: Path,
    *,
    output_root: Path,
    relative_root: Path | None,
    source_column: str,
    target_column: str,
    split_column: str,
    calibration_splits: tuple[str, ...],
    floor_column: str,
    low_quality_floor: float,
    high_quality_floor: float,
    quality_quantile: float,
    invert: bool,
) -> Path:
    df = pd.read_parquet(path)
    for column in (source_column, target_column):
        if column not in df.columns:
            raise ValueError(f"{path} is missing required column {column!r}.")

    target = pd.to_numeric(df[target_column], errors="coerce")
    if split_column in df.columns:
        calibration_mask = df[split_column].astype(str).isin(calibration_splits)
    else:
        calibration_mask = pd.Series(True, index=df.index)
    calibration = df[calibration_mask & target.notna()].copy()
    if calibration.empty:
        raise ValueError(f"{path} has no calibration rows for source-aware floors.")

    source_quality = (
        calibration.assign(_target=target.loc[calibration.index])
        .groupby(source_column, dropna=False)["_target"]
        .mean()
    )
    threshold = float(source_quality.quantile(quality_quantile))
    low_floor = high_quality_floor if invert else low_quality_floor
    high_floor = low_quality_floor if invert else high_quality_floor
    source_floor = {
        str(source): (high_floor if float(quality) >= threshold else low_floor)
        for source, quality in source_quality.items()
    }
    default_floor = high_floor if float(source_quality.mean()) >= threshold else low_floor

    augmented = df.copy()
    augmented[floor_column] = (
        augmented[source_column]
        .astype(str)
        .map(source_floor)
        .fillna(default_floor)
        .astype(float)
    )
    augmented[f"{floor_column}_source_quality"] = (
        augmented[source_column].astype(str).map(
            {str(source): float(value) for source, value in source_quality.items()}
        )
    )
    augmented[f"{floor_column}_quality_threshold"] = threshold

    if relative_root is not None:
        output_path = output_root / path.relative_to(relative_root)
    else:
        output_path = output_root / path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(output_path, index=False)
    return output_path


if __name__ == "__main__":
    main()
