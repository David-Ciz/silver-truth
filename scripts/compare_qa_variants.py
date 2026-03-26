#!/usr/bin/env python3
"""Compare QA-sensitive ablation outputs between two experiment variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _threshold_label(value: float) -> str:
    return f"{value:.2f}"


def _mean_from_csv(path: Path) -> Optional[dict[str, float]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "iou" not in df.columns or "f1" not in df.columns:
        return None
    if "split" in df.columns:
        test_df = df[df["split"].astype(str) == "test"]
        if not test_df.empty:
            df = test_df
    if df.empty:
        return None
    return {
        "iou": float(df["iou"].mean()),
        "f1": float(df["f1"].mean()),
        "count": float(len(df)),
    }


def _mean_from_parquet(path: Path) -> Optional[dict[str, float]]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "iou" not in df.columns or "f1" not in df.columns or df.empty:
        return None
    return {
        "iou": float(df["iou"].mean()),
        "f1": float(df["f1"].mean()),
        "count": float(len(df)),
    }


def _locate_ensemble_eval(mode_dir: Path) -> Optional[Path]:
    candidates = sorted(
        path
        for path in mode_dir.glob("*_set-test.parquet")
        if not path.name.endswith("_cell.parquet")
    )
    if not candidates:
        return None
    return candidates[-1]


def _load_setup_result(
    *,
    paper_runs_root: Path,
    dataset: str,
    crop_tag: str,
    variant: str,
    fold: str,
    setup_key: str,
    full_pipeline_model: str,
) -> Optional[dict[str, float | str]]:
    variant_root = paper_runs_root / "ablation" / dataset / crop_tag / variant / fold

    if setup_key == "qa_only":
        metrics = _mean_from_csv(variant_root / "qa_only" / "fullimage_eval.csv")
    elif setup_key.startswith("full_pipeline_t"):
        metrics = _mean_from_csv(
            variant_root / setup_key / full_pipeline_model / "fullimage_eval.csv"
        )
    elif setup_key.startswith("ensemble_qa_t"):
        parquet_path = _locate_ensemble_eval(variant_root / setup_key)
        metrics = _mean_from_parquet(parquet_path) if parquet_path else None
    else:
        raise ValueError(f"Unsupported setup key: {setup_key}")

    if metrics is None:
        return None

    return {
        "dataset": dataset,
        "crop_tag": crop_tag,
        "variant": variant,
        "fold": fold,
        "setup_key": setup_key,
        **metrics,
    }


def _build_setup_keys(thresholds: Iterable[float]) -> list[str]:
    keys = ["qa_only"]
    for threshold in thresholds:
        tag = _threshold_label(threshold)
        keys.append(f"full_pipeline_t{tag}")
    for threshold in thresholds:
        tag = _threshold_label(threshold)
        keys.append(f"ensemble_qa_t{tag}")
    return keys


def _best_by_family(
    rows_df: pd.DataFrame,
    *,
    variant: str,
    family_prefix: str,
) -> pd.DataFrame:
    subset = rows_df[
        (rows_df["variant"] == variant) & (rows_df["setup_key"].str.startswith(family_prefix))
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=["variant", "fold", "setup_key", "iou", "f1"])

    best_rows = (
        subset.sort_values(["fold", "iou", "f1", "setup_key"], ascending=[True, False, False, True])
        .groupby("fold", as_index=False)
        .first()
    )
    return best_rows[["variant", "fold", "setup_key", "iou", "f1"]]


def _format_float(value: object) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.4f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["_No rows found._"]

    table_df = df[columns].copy()
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in table_df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(_format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(
    *,
    paper_runs_root: Path,
    dataset: str,
    crop_tag: str,
    baseline_variant: str,
    candidate_variant: str,
    folds: list[str],
    thresholds: list[float],
    full_pipeline_model: str,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, float | str]] = []
    for variant in (baseline_variant, candidate_variant):
        for fold in folds:
            for setup_key in _build_setup_keys(thresholds):
                result = _load_setup_result(
                    paper_runs_root=paper_runs_root,
                    dataset=dataset,
                    crop_tag=crop_tag,
                    variant=variant,
                    fold=fold,
                    setup_key=setup_key,
                    full_pipeline_model=full_pipeline_model,
                )
                if result is not None:
                    rows.append(result)

    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        raise FileNotFoundError("No comparable QA result rows were found.")

    pivot_df = (
        rows_df.pivot_table(
            index=["fold", "setup_key"],
            columns="variant",
            values=["iou", "f1"],
            aggfunc="first",
        )
        .sort_index()
    )

    comparison_rows = []
    for (fold, setup_key), _ in pivot_df.iterrows():
        row = {"fold": fold, "setup_key": setup_key}
        baseline_iou = pivot_df.loc[(fold, setup_key), ("iou", baseline_variant)] if ("iou", baseline_variant) in pivot_df.columns else pd.NA
        baseline_f1 = pivot_df.loc[(fold, setup_key), ("f1", baseline_variant)] if ("f1", baseline_variant) in pivot_df.columns else pd.NA
        candidate_iou = pivot_df.loc[(fold, setup_key), ("iou", candidate_variant)] if ("iou", candidate_variant) in pivot_df.columns else pd.NA
        candidate_f1 = pivot_df.loc[(fold, setup_key), ("f1", candidate_variant)] if ("f1", candidate_variant) in pivot_df.columns else pd.NA
        row[f"{baseline_variant}_iou"] = baseline_iou
        row[f"{candidate_variant}_iou"] = candidate_iou
        row["delta_iou"] = (
            float(candidate_iou) - float(baseline_iou)
            if pd.notna(baseline_iou) and pd.notna(candidate_iou)
            else pd.NA
        )
        row[f"{baseline_variant}_f1"] = baseline_f1
        row[f"{candidate_variant}_f1"] = candidate_f1
        row["delta_f1"] = (
            float(candidate_f1) - float(baseline_f1)
            if pd.notna(baseline_f1) and pd.notna(candidate_f1)
            else pd.NA
        )
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["fold", "setup_key"])

    best_full_baseline = _best_by_family(rows_df, variant=baseline_variant, family_prefix="full_pipeline_t")
    best_full_candidate = _best_by_family(rows_df, variant=candidate_variant, family_prefix="full_pipeline_t")
    best_ensemble_baseline = _best_by_family(rows_df, variant=baseline_variant, family_prefix="ensemble_qa_t")
    best_ensemble_candidate = _best_by_family(rows_df, variant=candidate_variant, family_prefix="ensemble_qa_t")

    best_full = best_full_baseline.merge(
        best_full_candidate,
        on="fold",
        how="outer",
        suffixes=(f"__{baseline_variant}", f"__{candidate_variant}"),
    )
    if not best_full.empty:
        best_full["delta_iou"] = (
            best_full[f"iou__{candidate_variant}"] - best_full[f"iou__{baseline_variant}"]
        )

    best_ensemble = best_ensemble_baseline.merge(
        best_ensemble_candidate,
        on="fold",
        how="outer",
        suffixes=(f"__{baseline_variant}", f"__{candidate_variant}"),
    )
    if not best_ensemble.empty:
        best_ensemble["delta_iou"] = (
            best_ensemble[f"iou__{candidate_variant}"] - best_ensemble[f"iou__{baseline_variant}"]
        )

    lines = []
    lines.append("# QA Variant Comparison")
    lines.append("")
    lines.append(f"- dataset: `{dataset}`")
    lines.append(f"- crop: `{crop_tag}`")
    lines.append(f"- baseline variant: `{baseline_variant}`")
    lines.append(f"- candidate variant: `{candidate_variant}`")
    lines.append(f"- folds: `{', '.join(folds)}`")
    lines.append(f"- thresholds: `{', '.join(_threshold_label(t) for t in thresholds)}`")
    lines.append("")
    lines.append("## Per-Setup Comparison")
    lines.append("")
    lines.extend(
        _markdown_table(
            comparison_df,
            [
                "fold",
                "setup_key",
                f"{baseline_variant}_iou",
                f"{candidate_variant}_iou",
                "delta_iou",
                f"{baseline_variant}_f1",
                f"{candidate_variant}_f1",
                "delta_f1",
            ],
        )
    )
    lines.append("")
    lines.append("## Best Full Pipeline Per Fold")
    lines.append("")
    lines.extend(
        _markdown_table(
            best_full,
            [
                "fold",
                f"setup_key__{baseline_variant}",
                f"iou__{baseline_variant}",
                f"setup_key__{candidate_variant}",
                f"iou__{candidate_variant}",
                "delta_iou",
            ],
        )
    )
    lines.append("")
    lines.append("## Best Ensemble QA Per Fold")
    lines.append("")
    lines.extend(
        _markdown_table(
            best_ensemble,
            [
                "fold",
                f"setup_key__{baseline_variant}",
                f"iou__{baseline_variant}",
                f"setup_key__{candidate_variant}",
                f"iou__{candidate_variant}",
                "delta_iou",
            ],
        )
    )

    return comparison_df, "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-runs-root",
        type=Path,
        default=Path("data/paper_runs"),
        help="Root directory containing ablation outputs.",
    )
    parser.add_argument("--dataset", default="BF-C2DL-HSC")
    parser.add_argument("--crop-tag", default="sz64")
    parser.add_argument("--baseline-variant", default="baseline")
    parser.add_argument("--candidate-variant", required=True)
    parser.add_argument(
        "--folds",
        default="fold-1,fold-2",
        help="Comma-separated folds to compare.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.50,0.60,0.70,0.75",
        help="Comma-separated QA thresholds.",
    )
    parser.add_argument(
        "--full-pipeline-model",
        default="simple",
        help="Fusion model directory name used for full_pipeline results.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to save the markdown report.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the long-form comparison CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = [fold.strip() for fold in args.folds.split(",") if fold.strip()]
    thresholds = [
        float(raw.strip()) for raw in args.thresholds.split(",") if raw.strip()
    ]

    comparison_df, report_text = build_report(
        paper_runs_root=args.paper_runs_root,
        dataset=args.dataset,
        crop_tag=args.crop_tag,
        baseline_variant=args.baseline_variant,
        candidate_variant=args.candidate_variant,
        folds=folds,
        thresholds=thresholds,
        full_pipeline_model=args.full_pipeline_model,
    )

    print(report_text)

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(report_text)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
