#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FILENAME_PATTERN = re.compile(
    r"(?P<variant>.+?)_qa_predictions_(?P<fold>fold-[0-9]+|mixed)\.xlsx$"
)
CELL_ID_PATTERN = re.compile(
    r"^c(?P<campaign>\d+)_t(?P<time>\d+)_(?P<competitor>.+)_(?P<label>\d+)$"
)
TRUE_BINS = [-1e-9, 0.0, 0.25, 0.5, 0.75, 0.9, 1.01]
TRUE_BIN_LABELS = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,0.9]", "(0.9,1]"]


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {}
    return {
        "r2_score": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_correlation": float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman_correlation": float(stats.spearmanr(y_true, y_pred)[0]),
        "prediction_spread_ratio": float(np.std(y_pred) / np.std(y_true))
        if np.std(y_true) > 0
        else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize QA prediction Excel bundles by split composition, timepoint drift, "
            "and residual behavior."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Directory to scan for *_qa_predictions_*.xlsx files.",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        nargs="+",
        help="Explicit QA prediction Excel files to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSV summaries and the markdown report will be written.",
    )
    return parser.parse_args()


def _discover_excel_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.excel:
        paths.extend(path.resolve() for path in args.excel)
    if args.input_root:
        paths.extend(
            path.resolve()
            for path in sorted(args.input_root.rglob("*_qa_predictions_*.xlsx"))
        )
    deduped = sorted(set(paths))
    if not deduped:
        raise FileNotFoundError("No QA prediction Excel files were provided or found.")
    missing = [path for path in deduped if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing QA prediction Excel files:\n{missing_text}")
    return deduped


def _parse_file_metadata(path: Path) -> dict[str, str]:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return {"variant": path.stem, "fold": "unknown"}
    return match.groupdict()


def _load_split_df(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=split).copy()
    if "Jaccard index" not in df.columns or "Predicted Jaccard index" not in df.columns:
        raise ValueError(
            f"{path} sheet {split} is missing expected QA columns."
        )
    meta = df["cell_id"].astype(str).str.extract(CELL_ID_PATTERN)
    if meta.isna().any().any():
        raise ValueError(f"Could not parse one or more cell IDs in {path} sheet {split}.")
    meta["campaign"] = meta["campaign"].astype(int)
    meta["time"] = meta["time"].astype(int)
    meta["label"] = meta["label"].astype(int)
    for column in meta.columns:
        df[column] = meta[column]
    df["split"] = split
    df["residual"] = df["Predicted Jaccard index"] - df["Jaccard index"]
    return df


def _campaign_summary(df: pd.DataFrame) -> str:
    counts = (
        df.groupby("campaign")
        .size()
        .sort_index()
        .items()
    )
    return ", ".join(f"{campaign}:{count}" for campaign, count in counts)


def _split_summary_rows(path: Path) -> list[dict[str, object]]:
    meta = _parse_file_metadata(path)
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        df = _load_split_df(path, split)
        metrics = calculate_regression_metrics(
            df["Jaccard index"].to_numpy(),
            df["Predicted Jaccard index"].to_numpy(),
        )
        rows.append(
            {
                "excel_name": path.name,
                "variant": meta["variant"],
                "fold": meta["fold"],
                "split": split,
                "n_samples": int(len(df)),
                "campaign_counts": _campaign_summary(df),
                "n_timepoints": int(df["time"].nunique()),
                "time_min": int(df["time"].min()),
                "time_max": int(df["time"].max()),
                "true_mean": float(df["Jaccard index"].mean()),
                "pred_mean": float(df["Predicted Jaccard index"].mean()),
                "mean_residual": float(df["residual"].mean()),
                "pearson": float(metrics["pearson_correlation"]),
                "spearman": float(metrics["spearman_correlation"]),
                "r2": float(metrics["r2_score"]),
                "mae": float(metrics["mae"]),
                "spread_ratio": float(metrics["prediction_spread_ratio"]),
                "zero_pct": float((df["Jaccard index"] == 0).mean() * 100.0),
                "low_lt_05_pct": float((df["Jaccard index"] < 0.5).mean() * 100.0),
                "good_ge_075_pct": float((df["Jaccard index"] >= 0.75).mean() * 100.0),
                "high_ge_09_pct": float((df["Jaccard index"] >= 0.9).mean() * 100.0),
            }
        )
    return rows


def _timepoint_summary_rows(path: Path) -> list[dict[str, object]]:
    meta = _parse_file_metadata(path)
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        df = _load_split_df(path, split)
        grouped = (
            df.groupby(["campaign", "time"], as_index=False)
            .agg(
                n_samples=("cell_id", "size"),
                true_mean=("Jaccard index", "mean"),
                pred_mean=("Predicted Jaccard index", "mean"),
                mean_residual=("residual", "mean"),
                zero_pct=("Jaccard index", lambda series: (series == 0).mean() * 100.0),
                good_ge_075_pct=(
                    "Jaccard index",
                    lambda series: (series >= 0.75).mean() * 100.0,
                ),
            )
            .sort_values(["campaign", "time"])
        )
        for row in grouped.to_dict(orient="records"):
            rows.append(
                {
                    "excel_name": path.name,
                    "variant": meta["variant"],
                    "fold": meta["fold"],
                    "split": split,
                    **row,
                }
            )
    return rows


def _residual_bin_rows(path: Path) -> list[dict[str, object]]:
    meta = _parse_file_metadata(path)
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        df = _load_split_df(path, split)
        df["true_bin"] = pd.cut(
            df["Jaccard index"],
            bins=TRUE_BINS,
            labels=TRUE_BIN_LABELS,
            include_lowest=True,
            right=True,
        )
        grouped = (
            df.groupby("true_bin", observed=False, as_index=False)
            .agg(
                n_samples=("cell_id", "size"),
                true_mean=("Jaccard index", "mean"),
                pred_mean=("Predicted Jaccard index", "mean"),
                mean_residual=("residual", "mean"),
            )
        )
        for row in grouped.to_dict(orient="records"):
            rows.append(
                {
                    "excel_name": path.name,
                    "variant": meta["variant"],
                    "fold": meta["fold"],
                    "split": split,
                    **row,
                }
            )
    return rows


def _format_metric(value: object, digits: int = 4) -> object:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _build_highlights(
    split_df: pd.DataFrame, timepoint_df: pd.DataFrame, residual_df: pd.DataFrame
) -> list[str]:
    notes: list[str] = []
    for fold in sorted(split_df["fold"].unique()):
        fold_split = split_df[split_df["fold"] == fold].set_index("split")
        if {"validation", "test"} - set(fold_split.index):
            continue
        validation = fold_split.loc["validation"]
        test = fold_split.loc["test"]
        train = fold_split.loc["train"]

        if int(validation["n_timepoints"]) == 1:
            notes.append(
                f"`{fold}` validation is a single timepoint "
                f"(campaigns `{validation['campaign_counts']}`, time {int(validation['time_min'])})."
            )

        good_gap = float(validation["good_ge_075_pct"] - test["good_ge_075_pct"])
        zero_gap = float(validation["zero_pct"] - test["zero_pct"])
        if abs(good_gap) >= 20.0 or abs(zero_gap) >= 20.0:
            notes.append(
                f"`{fold}` validation is not representative of test: "
                f"good-crop gap {good_gap:+.1f} pts, zero-IoU gap {zero_gap:+.1f} pts."
            )

        if str(train["campaign_counts"]) != str(test["campaign_counts"]):
            notes.append(
                f"`{fold}` trains on campaign `{train['campaign_counts']}` but tests on "
                f"`{test['campaign_counts']}`, so the QA scorer sees cross-campaign shift."
            )

        fold_test_bins = residual_df[
            (residual_df["fold"] == fold) & (residual_df["split"] == "test")
        ].dropna(subset=["mean_residual"])
        if not fold_test_bins.empty:
            worst = fold_test_bins.iloc[
                fold_test_bins["mean_residual"].abs().idxmax()
                - fold_test_bins.index.min()
            ]
            notes.append(
                f"`{fold}` test shows strongest calibration error in true-score bin "
                f"`{worst['true_bin']}` with mean residual {float(worst['mean_residual']):+.3f}."
            )

        val_timepoints = timepoint_df[
            (timepoint_df["fold"] == fold) & (timepoint_df["split"] == "validation")
        ]
        if not val_timepoints.empty:
            row = val_timepoints.iloc[0]
            notes.append(
                f"`{fold}` validation timepoint `c{int(row['campaign']):02d}_t{int(row['time']):03d}` "
                f"has true mean {float(row['true_mean']):.3f} and zero-IoU rate "
                f"{float(row['zero_pct']):.1f}%."
            )
    return notes


def build_markdown(
    *,
    excel_paths: Iterable[Path],
    split_df: pd.DataFrame,
    timepoint_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines: list[str] = [
        "# QA Prediction Bundle Analysis",
        "",
        "This report summarizes saved QA prediction Excel bundles with an emphasis on:",
        "",
        "- split composition drift",
        "- timepoint-level dataset shift",
        "- calibration errors by true-score range",
        "",
        "Source files:",
        "",
    ]
    for path in excel_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "Generated artifacts:",
            "",
            f"- `{output_dir / 'split_summary.csv'}`",
            f"- `{output_dir / 'timepoint_summary.csv'}`",
            f"- `{output_dir / 'residual_bin_summary.csv'}`",
            "",
            "## Global Highlights",
            "",
        ]
    )
    for note in _build_highlights(split_df, timepoint_df, residual_df):
        lines.append(f"- {note}")

    for fold in sorted(split_df["fold"].unique()):
        lines.extend([f"", f"## {fold}", ""])
        fold_split = split_df[split_df["fold"] == fold][
            [
                "split",
                "n_samples",
                "campaign_counts",
                "n_timepoints",
                "time_min",
                "time_max",
                "true_mean",
                "pred_mean",
                "mean_residual",
                "pearson",
                "spearman",
                "r2",
                "mae",
                "spread_ratio",
                "zero_pct",
                "good_ge_075_pct",
            ]
        ].copy()
        for column in fold_split.columns:
            fold_split[column] = fold_split[column].map(_format_metric)
        lines.append("### Split Summary")
        lines.append("")
        lines.append(fold_split.to_markdown(index=False))
        lines.append("")

        fold_timepoints = timepoint_df[timepoint_df["fold"] == fold][
            [
                "split",
                "campaign",
                "time",
                "n_samples",
                "true_mean",
                "pred_mean",
                "mean_residual",
                "zero_pct",
                "good_ge_075_pct",
            ]
        ].copy()
        for column in fold_timepoints.columns:
            fold_timepoints[column] = fold_timepoints[column].map(_format_metric)
        lines.append("### Timepoint Summary")
        lines.append("")
        lines.append(fold_timepoints.to_markdown(index=False))
        lines.append("")

        fold_bins = residual_df[
            (residual_df["fold"] == fold) & (residual_df["split"] == "test")
        ][["true_bin", "n_samples", "true_mean", "pred_mean", "mean_residual"]].copy()
        for column in fold_bins.columns:
            fold_bins[column] = fold_bins[column].map(_format_metric)
        lines.append("### Test Residuals By True-Score Bin")
        lines.append("")
        lines.append(fold_bins.to_markdown(index=False))
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    excel_paths = _discover_excel_paths(args)
    split_rows: list[dict[str, object]] = []
    timepoint_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []

    for path in excel_paths:
        split_rows.extend(_split_summary_rows(path))
        timepoint_rows.extend(_timepoint_summary_rows(path))
        residual_rows.extend(_residual_bin_rows(path))

    split_df = pd.DataFrame(split_rows).sort_values(["fold", "split"]).reset_index(drop=True)
    timepoint_df = pd.DataFrame(timepoint_rows).sort_values(
        ["fold", "split", "campaign", "time"]
    ).reset_index(drop=True)
    residual_df = pd.DataFrame(residual_rows).sort_values(
        ["fold", "split", "true_bin"]
    ).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output_dir / "split_summary.csv", index=False)
    timepoint_df.to_csv(args.output_dir / "timepoint_summary.csv", index=False)
    residual_df.to_csv(args.output_dir / "residual_bin_summary.csv", index=False)

    report = build_markdown(
        excel_paths=excel_paths,
        split_df=split_df,
        timepoint_df=timepoint_df,
        residual_df=residual_df,
        output_dir=args.output_dir,
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
