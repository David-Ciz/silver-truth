from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def generate_ablation_diagnostics_report(
    *,
    qa_metrics_csv: Path,
    qa_filtering_csv: Path,
    ablation_dir: Path,
    dataset: str,
    crop_tag: str,
    variant: str,
    split_name: str,
    default_threshold: float,
) -> dict[str, pd.DataFrame]:
    qa_metrics = _load_qa_metrics(qa_metrics_csv)
    qa_filtering = _load_qa_filtering(qa_filtering_csv)
    scan_root = _discover_ablation_scan_root(ablation_dir)
    all_ablation_summary = _summarize_ablation_outputs(scan_root)
    ablation_summary = all_ablation_summary[
        all_ablation_summary["fold"].astype(str) == split_name
    ].reset_index(drop=True)
    cross_fold_summary = _build_cross_fold_summary(all_ablation_summary)
    flags = _build_flags(
        qa_metrics=qa_metrics,
        qa_filtering=qa_filtering,
        ablation_summary=ablation_summary,
        cross_fold_summary=cross_fold_summary,
        default_threshold=default_threshold,
    )
    overview = _build_overview(
        qa_metrics=qa_metrics,
        qa_filtering=qa_filtering,
        ablation_summary=ablation_summary,
        cross_fold_summary=cross_fold_summary,
        flags=flags,
        dataset=dataset,
        crop_tag=crop_tag,
        variant=variant,
        split_name=split_name,
        default_threshold=default_threshold,
    )
    markdown = _render_markdown_report(
        overview=overview,
        qa_metrics=qa_metrics,
        qa_filtering=qa_filtering,
        ablation_summary=ablation_summary,
        cross_fold_summary=cross_fold_summary,
        flags=flags,
        default_threshold=default_threshold,
    )
    return {
        "overview": overview,
        "qa_regression_summary": qa_metrics,
        "qa_threshold_summary": qa_filtering,
        "ablation_method_summary": ablation_summary,
        "cross_fold_method_summary": cross_fold_summary,
        "diagnostic_flags": flags,
        "report_markdown": pd.DataFrame([{"markdown": markdown}]),
    }


def write_ablation_diagnostics_report(
    output_dir: Path, bundle: dict[str, pd.DataFrame]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "overview": output_dir / "overview.csv",
        "qa_regression_summary": output_dir / "qa_regression_summary.csv",
        "qa_threshold_summary": output_dir / "qa_threshold_summary.csv",
        "ablation_method_summary": output_dir / "ablation_method_summary.csv",
        "cross_fold_method_summary": output_dir / "cross_fold_method_summary.csv",
        "diagnostic_flags": output_dir / "diagnostic_flags.csv",
        "report_markdown": output_dir / "ablation_diagnostics_report.md",
    }

    for key, path in paths.items():
        if key == "report_markdown":
            path.write_text(
                str(bundle["report_markdown"].iloc[0]["markdown"]), encoding="utf-8"
            )
        else:
            bundle[key].to_csv(path, index=False)

    return paths


def _load_qa_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col.startswith("Unnamed"):
        if "split" not in df.columns:
            df = df.rename(columns={first_col: "split"})
        else:
            df["split"] = df["split"].fillna(df[first_col])
            df = df.drop(columns=[first_col])
    if "split" not in df.columns:
        raise ValueError(f"QA metrics CSV is missing a split column: {path}")

    df = df.copy()
    df["split"] = df["split"].astype(str)
    numeric_columns = [column for column in df.columns if column != "split"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    preferred_columns = [
        "split",
        "n_samples",
        "r2_score",
        "mae",
        "rmse",
        "pearson_correlation",
        "spearman_correlation",
        "mean_true",
        "mean_pred",
        "mean_residual",
        "prediction_spread_ratio",
    ]
    present = [column for column in preferred_columns if column in df.columns]
    remaining = [column for column in df.columns if column not in present]
    split_order = {"train": 0, "validation": 1, "test": 2, "combined": 3}
    df = df[present + remaining].copy()
    df["_split_order"] = df["split"].map(split_order).fillna(99)
    return (
        df.sort_values(["_split_order", "split"])
        .drop(columns=["_split_order"])
        .reset_index(drop=True)
    )


def _load_qa_filtering(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    if "split" not in df.columns:
        df["split"] = "test"

    for column in (
        "actual_good_pct",
        "predicted_good_pct",
        "support_gap_pct",
        "support_ratio",
    ):
        if column not in df.columns:
            if column == "actual_good_pct":
                df[column] = 100.0 * df["actual_good_count"] / df["n_samples"]
            elif column == "predicted_good_pct":
                df[column] = 100.0 * df["kept_count"] / df["n_samples"]
            elif column == "support_gap_pct":
                df[column] = df["predicted_good_pct"] - df["actual_good_pct"]
            elif column == "support_ratio":
                df[column] = df["kept_count"] / df["actual_good_count"].clip(lower=1.0)

    sort_columns = [column for column in ["split", "threshold"] if column in df.columns]
    return df.sort_values(sort_columns).reset_index(drop=True)


def _summarize_ablation_outputs(ablation_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for csv_path in sorted(ablation_dir.rglob("fullimage_eval.csv")):
        record = _summarize_result_table(csv_path)
        if record is None:
            continue
        records.append(record)

    for parquet_path in sorted(ablation_dir.rglob("*_set-test.parquet")):
        record = _summarize_result_table(parquet_path)
        if record is None:
            continue
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "method_key",
                "fold",
                "family",
                "uses_qa",
                "threshold",
                "fusion_model",
                "n_images",
                "mean_iou",
                "mean_f1",
                "p05_iou",
                "min_iou",
                "bad_image_count",
                "bad_image_pct",
                "mean_cells_placed",
                "reference_method_key",
                "delta_iou_vs_reference",
                "delta_f1_vs_reference",
                "delta_cells_placed_vs_reference",
                "source_path",
            ]
        )

    df = pd.DataFrame.from_records(records)
    reference_keys: list[str | None] = []
    delta_ious: list[float] = []
    delta_f1s: list[float] = []
    delta_cells: list[float] = []
    for _, row in df.iterrows():
        reference_key: str | None = None
        reference_row = pd.DataFrame()
        if row["family"] == "full_pipeline":
            reference_row = df[
                (df["fold"] == row["fold"])
                & (df["family"] == "fusion_only")
                & (df["fusion_model"] == row["fusion_model"])
            ]
        elif row["family"] in {"ensemble_qa", "ensemble_qa_retrained"}:
            reference_row = df[
                (df["fold"] == row["fold"]) & (df["family"] == "ensemble_only")
            ]

        if not reference_row.empty:
            reference_key = str(reference_row.iloc[0]["method_key"])

        reference_keys.append(reference_key)
        if reference_key is None:
            delta_ious.append(float("nan"))
            delta_f1s.append(float("nan"))
            delta_cells.append(float("nan"))
            continue

        if reference_row.empty:
            delta_ious.append(float("nan"))
            delta_f1s.append(float("nan"))
            delta_cells.append(float("nan"))
            continue

        delta_ious.append(float(row["mean_iou"] - reference_row.iloc[0]["mean_iou"]))
        delta_f1s.append(float(row["mean_f1"] - reference_row.iloc[0]["mean_f1"]))
        delta_cells.append(
            float(row["mean_cells_placed"] - reference_row.iloc[0]["mean_cells_placed"])
        )

    df["reference_method_key"] = reference_keys
    df["delta_iou_vs_reference"] = delta_ious
    df["delta_f1_vs_reference"] = delta_f1s
    df["delta_cells_placed_vs_reference"] = delta_cells

    return df.sort_values(
        ["fold", "mean_iou", "family", "method_key"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def _summarize_result_table(path: Path) -> dict[str, Any] | None:
    relative = path.parts
    fold = _parse_fold(relative)
    if "qa_only" in relative:
        method_key = "qa_only__top1"
        family = "qa_only"
        threshold = None
        fusion_model = None
        uses_qa = True
    elif "fusion_only" in relative:
        idx = relative.index("fusion_only")
        if len(relative) <= idx + 2:
            return None
        fusion_model = relative[idx + 1]
        method_key = f"fusion_only__{fusion_model}"
        family = "fusion_only"
        threshold = None
        uses_qa = False
    else:
        threshold_dir = next(
            (
                part
                for part in relative
                if part.startswith("full_pipeline_t")
                or part.startswith("ensemble_qa_t")
                or part.startswith("ensemble_qa_retrained_t")
                or part == "ensemble_only"
            ),
            None,
        )
        if threshold_dir is None:
            return None
        if threshold_dir == "ensemble_only":
            method_key = "ensemble_only"
            family = "ensemble_only"
            threshold = None
            fusion_model = None
            uses_qa = False
        elif threshold_dir.startswith("full_pipeline_t"):
            idx = relative.index(threshold_dir)
            if len(relative) <= idx + 2:
                return None
            fusion_model = relative[idx + 1]
            method_key = f"{threshold_dir}__{fusion_model}"
            family = "full_pipeline"
            threshold = _parse_threshold(threshold_dir)
            uses_qa = True
        elif threshold_dir.startswith("ensemble_qa_retrained_t"):
            method_key = threshold_dir
            family = "ensemble_qa_retrained"
            threshold = _parse_threshold(threshold_dir)
            fusion_model = None
            uses_qa = True
        elif threshold_dir.startswith("ensemble_qa_t"):
            method_key = threshold_dir
            family = "ensemble_qa"
            threshold = _parse_threshold(threshold_dir)
            fusion_model = None
            uses_qa = True
        else:
            return None

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    if "iou" not in df.columns or "f1" not in df.columns:
        return None

    return {
        "method_key": method_key,
        "fold": fold,
        "family": family,
        "uses_qa": uses_qa,
        "threshold": threshold,
        "fusion_model": fusion_model,
        "n_images": int(len(df)),
        "mean_iou": float(pd.to_numeric(df["iou"], errors="coerce").mean()),
        "mean_f1": float(pd.to_numeric(df["f1"], errors="coerce").mean()),
        "p05_iou": float(pd.to_numeric(df["iou"], errors="coerce").quantile(0.05)),
        "min_iou": float(pd.to_numeric(df["iou"], errors="coerce").min()),
        "bad_image_count": int(
            (pd.to_numeric(df["iou"], errors="coerce") < 0.50).sum()
        ),
        "bad_image_pct": float(
            100.0 * (pd.to_numeric(df["iou"], errors="coerce") < 0.50).mean()
        ),
        "mean_cells_placed": float(
            pd.to_numeric(df.get("cells_placed"), errors="coerce").mean()
        )
        if "cells_placed" in df.columns
        else float("nan"),
        "source_path": str(path),
    }


def _parse_threshold(token: str) -> float | None:
    match = re.search(r"_t(\d+(?:\.\d+)?)", token)
    if not match:
        return None
    return float(match.group(1))


def _parse_fold(path_parts: tuple[str, ...]) -> str:
    for part in path_parts:
        if re.fullmatch(r"fold-\d+", part) or part == "mixed":
            return part
    return "unknown"


def _discover_ablation_scan_root(ablation_dir: Path) -> Path:
    parent = ablation_dir.parent
    sibling_folds = [
        child
        for child in parent.iterdir()
        if child.is_dir() and re.fullmatch(r"fold-\d+|mixed", child.name)
    ]
    if sibling_folds:
        return parent
    return ablation_dir


def _build_cross_fold_summary(ablation_summary: pd.DataFrame) -> pd.DataFrame:
    if ablation_summary.empty or "fold" not in ablation_summary.columns:
        return pd.DataFrame()

    valid = ablation_summary[
        ablation_summary["fold"].astype(str).str.startswith("fold-")
    ]
    if valid.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for method_key, group in valid.groupby("method_key"):
        if group["fold"].nunique() < 2:
            continue
        group = group.sort_values("fold")
        records.append(
            {
                "method_key": method_key,
                "family": group.iloc[0]["family"],
                "uses_qa": bool(group.iloc[0]["uses_qa"]),
                "n_folds": int(group["fold"].nunique()),
                "mean_iou_across_folds": float(group["mean_iou"].mean()),
                "fold_iou_gap": float(
                    group["mean_iou"].max() - group["mean_iou"].min()
                ),
                "mean_f1_across_folds": float(group["mean_f1"].mean()),
                "fold_f1_gap": float(group["mean_f1"].max() - group["mean_f1"].min()),
                "worst_fold_iou": float(group["mean_iou"].min()),
                "best_fold_iou": float(group["mean_iou"].max()),
            }
        )

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["fold_iou_gap", "mean_iou_across_folds"], ascending=[False, False]
        )
        .reset_index(drop=True)
    )


def _build_flags(
    *,
    qa_metrics: pd.DataFrame,
    qa_filtering: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    cross_fold_summary: pd.DataFrame,
    default_threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    metrics_by_split = qa_metrics.set_index("split", drop=False)
    test_row = (
        metrics_by_split.loc["test"] if "test" in metrics_by_split.index else None
    )
    train_row = (
        metrics_by_split.loc["train"] if "train" in metrics_by_split.index else None
    )

    default_filter_row = _closest_threshold_row(qa_filtering, default_threshold)

    _append_flag(
        records,
        code="qa_test_pearson_low",
        severity="warning",
        triggered=test_row is not None
        and float(test_row.get("pearson_correlation", float("nan"))) < 0.40,
        detail=(
            f"test Pearson={float(test_row['pearson_correlation']):.4f} < 0.40"
            if test_row is not None and pd.notna(test_row.get("pearson_correlation"))
            else "test Pearson unavailable"
        ),
    )
    _append_flag(
        records,
        code="qa_test_spearman_low",
        severity="warning",
        triggered=test_row is not None
        and float(test_row.get("spearman_correlation", float("nan"))) < 0.30,
        detail=(
            f"test Spearman={float(test_row['spearman_correlation']):.4f} < 0.30"
            if test_row is not None and pd.notna(test_row.get("spearman_correlation"))
            else "test Spearman unavailable"
        ),
    )
    _append_flag(
        records,
        code="qa_test_r2_negative",
        severity="warning",
        triggered=test_row is not None
        and float(test_row.get("r2_score", float("nan"))) < 0.0,
        detail=(
            f"test R2={float(test_row['r2_score']):.4f} < 0"
            if test_row is not None and pd.notna(test_row.get("r2_score"))
            else "test R2 unavailable"
        ),
    )
    _append_flag(
        records,
        code="qa_train_test_gap_large",
        severity="warning",
        triggered=train_row is not None
        and test_row is not None
        and pd.notna(train_row.get("pearson_correlation"))
        and pd.notna(test_row.get("pearson_correlation"))
        and float(train_row["pearson_correlation"] - test_row["pearson_correlation"])
        > 0.30,
        detail=(
            f"train-test Pearson gap="
            f"{float(train_row['pearson_correlation'] - test_row['pearson_correlation']):.4f}"
            if train_row is not None
            and test_row is not None
            and pd.notna(train_row.get("pearson_correlation"))
            and pd.notna(test_row.get("pearson_correlation"))
            else "train/test Pearson unavailable"
        ),
    )
    _append_flag(
        records,
        code="qa_test_bias_large_negative",
        severity="warning",
        triggered=test_row is not None
        and float(test_row.get("mean_residual", float("nan"))) <= -0.10,
        detail=(
            f"test mean_residual={float(test_row['mean_residual']):.4f} <= -0.10"
            if test_row is not None and pd.notna(test_row.get("mean_residual"))
            else "test bias unavailable"
        ),
    )
    _append_flag(
        records,
        code="qa_test_predictions_compressed",
        severity="warning",
        triggered=test_row is not None
        and float(test_row.get("prediction_spread_ratio", float("nan"))) < 0.60,
        detail=(
            f"test spread ratio={float(test_row['prediction_spread_ratio']):.4f} < 0.60"
            if test_row is not None
            and pd.notna(test_row.get("prediction_spread_ratio"))
            else "test spread ratio unavailable"
        ),
    )
    _append_flag(
        records,
        code="default_threshold_underkeeps_good_samples",
        severity="warning",
        triggered=default_filter_row is not None
        and float(default_filter_row["support_gap_pct"]) <= -20.0,
        detail=(
            f"t={float(default_filter_row['threshold']):.2f}: predicted good "
            f"{float(default_filter_row['predicted_good_pct']):.2f}% vs actual "
            f"{float(default_filter_row['actual_good_pct']):.2f}%"
            if default_filter_row is not None
            else "default threshold missing"
        ),
    )
    _append_flag(
        records,
        code="default_threshold_recall_low",
        severity="warning",
        triggered=default_filter_row is not None
        and float(default_filter_row["recall"]) < 0.50,
        detail=(
            f"t={float(default_filter_row['threshold']):.2f}: recall={float(default_filter_row['recall']):.4f}"
            if default_filter_row is not None
            else "default threshold missing"
        ),
    )
    _append_flag(
        records,
        code="default_threshold_filtered_pct_high",
        severity="warning",
        triggered=default_filter_row is not None
        and float(default_filter_row["filtered_pct"]) > 75.0,
        detail=(
            f"t={float(default_filter_row['threshold']):.2f}: filtered_pct={float(default_filter_row['filtered_pct']):.2f}%"
            if default_filter_row is not None
            else "default threshold missing"
        ),
    )

    best_full_pipeline = _best_method(ablation_summary, family="full_pipeline")
    best_qa_method = _best_method(ablation_summary, uses_qa=True)
    best_non_qa = _best_method(ablation_summary, uses_qa=False)
    best_threshold_fragility = _threshold_fragility_row(ablation_summary)
    model_dependence = _model_dependence_row(ablation_summary)
    tail_risk_method = _tail_risk_row(ablation_summary)
    retrained_row = _retrained_value_row(ablation_summary, default_threshold)
    unstable_method = _fold_instability_row(cross_fold_summary)

    _append_flag(
        records,
        code="best_full_pipeline_below_default_threshold",
        severity="info",
        triggered=best_full_pipeline is not None
        and pd.notna(best_full_pipeline.get("threshold"))
        and float(best_full_pipeline["threshold"]) < default_threshold,
        detail=(
            f"best full_pipeline is {best_full_pipeline['method_key']} "
            f"(IoU={float(best_full_pipeline['mean_iou']):.4f})"
            if best_full_pipeline is not None
            else "no full_pipeline results"
        ),
    )
    _append_flag(
        records,
        code="qa_method_beats_non_qa_despite_validity_warnings",
        severity="info",
        triggered=best_qa_method is not None
        and best_non_qa is not None
        and float(best_qa_method["mean_iou"]) > float(best_non_qa["mean_iou"])
        and any(
            code in {"qa_test_pearson_low", "default_threshold_underkeeps_good_samples"}
            for code in {
                record["code"] for record in records if bool(record["triggered"])
            }
        ),
        detail=(
            f"best QA method {best_qa_method['method_key']} "
            f"(IoU={float(best_qa_method['mean_iou']):.4f}) beats best non-QA "
            f"{best_non_qa['method_key']} (IoU={float(best_non_qa['mean_iou']):.4f})"
            if best_qa_method is not None and best_non_qa is not None
            else "best-method comparison unavailable"
        ),
    )
    _append_flag(
        records,
        code="best_method_gain_below_min_effect_size",
        severity="info",
        triggered=best_qa_method is not None
        and pd.notna(best_qa_method.get("delta_iou_vs_reference"))
        and float(best_qa_method["delta_iou_vs_reference"]) < 0.01,
        detail=(
            f"{best_qa_method['method_key']} delta vs reference="
            f"{float(best_qa_method['delta_iou_vs_reference']):.4f}"
            if best_qa_method is not None
            else "best QA method unavailable"
        ),
    )
    _append_flag(
        records,
        code="threshold_response_brittle",
        severity="warning",
        triggered=best_threshold_fragility is not None
        and float(best_threshold_fragility["threshold_range_iou"]) >= 0.03,
        detail=(
            f"{best_threshold_fragility['family']} / "
            f"{best_threshold_fragility['fusion_model']} range across thresholds="
            f"{float(best_threshold_fragility['threshold_range_iou']):.4f}"
            if best_threshold_fragility is not None
            else "threshold fragility unavailable"
        ),
    )
    _append_flag(
        records,
        code="model_dependence_high",
        severity="info",
        triggered=model_dependence is not None
        and float(model_dependence["model_spread_iou"]) >= 0.03,
        detail=(
            f"best-vs-worst full_pipeline fuser spread at t="
            f"{float(model_dependence['threshold']):.2f} is "
            f"{float(model_dependence['model_spread_iou']):.4f}"
            if model_dependence is not None
            else "model dependence unavailable"
        ),
    )
    _append_flag(
        records,
        code="tail_failure_rate_high",
        severity="warning",
        triggered=tail_risk_method is not None
        and float(tail_risk_method["bad_image_pct"]) >= 10.0,
        detail=(
            f"{tail_risk_method['method_key']} has "
            f"{float(tail_risk_method['bad_image_pct']):.2f}% images with IoU < 0.50 "
            f"(p05={float(tail_risk_method['p05_iou']):.4f}, min={float(tail_risk_method['min_iou']):.4f})"
            if tail_risk_method is not None
            else "tail-risk unavailable"
        ),
    )
    _append_flag(
        records,
        code="cells_placed_shift_large",
        severity="info",
        triggered=best_qa_method is not None
        and pd.notna(best_qa_method.get("delta_cells_placed_vs_reference"))
        and abs(float(best_qa_method["delta_cells_placed_vs_reference"])) >= 0.50,
        detail=(
            f"{best_qa_method['method_key']} mean cells_placed shift vs reference="
            f"{float(best_qa_method['delta_cells_placed_vs_reference']):.4f}"
            if best_qa_method is not None
            else "cells_placed shift unavailable"
        ),
    )
    _append_flag(
        records,
        code="retrained_not_better_than_frozen",
        severity="warning",
        triggered=retrained_row is not None
        and float(retrained_row["delta_vs_best_baseline"]) <= 0.0,
        detail=(
            f"{retrained_row['method_key']} delta vs best frozen baseline="
            f"{float(retrained_row['delta_vs_best_baseline']):.4f}"
            if retrained_row is not None
            else "retrained comparison unavailable"
        ),
    )
    _append_flag(
        records,
        code="large_fold_gap",
        severity="warning",
        triggered=unstable_method is not None
        and float(unstable_method["fold_iou_gap"]) >= 0.03,
        detail=(
            f"{unstable_method['method_key']} fold IoU gap="
            f"{float(unstable_method['fold_iou_gap']):.4f}"
            if unstable_method is not None
            else "cross-fold summary unavailable"
        ),
    )

    return pd.DataFrame.from_records(records)


def _build_overview(
    *,
    qa_metrics: pd.DataFrame,
    qa_filtering: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    cross_fold_summary: pd.DataFrame,
    flags: pd.DataFrame,
    dataset: str,
    crop_tag: str,
    variant: str,
    split_name: str,
    default_threshold: float,
) -> pd.DataFrame:
    test_metrics = _first_split_row(qa_metrics, "test")
    default_filter = _closest_threshold_row(qa_filtering, default_threshold)
    best_overall = _best_method(ablation_summary)
    best_qa = _best_method(ablation_summary, uses_qa=True)
    best_non_qa = _best_method(ablation_summary, uses_qa=False)
    unstable_method = _fold_instability_row(cross_fold_summary)

    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "crop_tag": crop_tag,
                "variant": variant,
                "split_name": split_name,
                "default_threshold": default_threshold,
                "test_pearson": _series_value(test_metrics, "pearson_correlation"),
                "test_spearman": _series_value(test_metrics, "spearman_correlation"),
                "test_r2": _series_value(test_metrics, "r2_score"),
                "test_mae": _series_value(test_metrics, "mae"),
                "test_mean_residual": _series_value(test_metrics, "mean_residual"),
                "default_actual_good_pct": _series_value(
                    default_filter, "actual_good_pct"
                ),
                "default_predicted_good_pct": _series_value(
                    default_filter, "predicted_good_pct"
                ),
                "default_support_gap_pct": _series_value(
                    default_filter, "support_gap_pct"
                ),
                "default_recall": _series_value(default_filter, "recall"),
                "default_filtered_pct": _series_value(default_filter, "filtered_pct"),
                "best_overall_method": _series_value(best_overall, "method_key"),
                "best_overall_iou": _series_value(best_overall, "mean_iou"),
                "best_qa_method": _series_value(best_qa, "method_key"),
                "best_qa_iou": _series_value(best_qa, "mean_iou"),
                "best_non_qa_method": _series_value(best_non_qa, "method_key"),
                "best_non_qa_iou": _series_value(best_non_qa, "mean_iou"),
                "largest_fold_gap_method": _series_value(unstable_method, "method_key"),
                "largest_fold_gap_iou": _series_value(unstable_method, "fold_iou_gap"),
                "n_triggered_flags": int(flags["triggered"].fillna(False).sum())
                if not flags.empty
                else 0,
            }
        ]
    )


def _render_markdown_report(
    *,
    overview: pd.DataFrame,
    qa_metrics: pd.DataFrame,
    qa_filtering: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    cross_fold_summary: pd.DataFrame,
    flags: pd.DataFrame,
    default_threshold: float,
) -> str:
    lines = ["# Ablation Diagnostics", ""]

    if not overview.empty:
        row = overview.iloc[0]
        lines.extend(
            [
                f"- Dataset: `{row['dataset']}`",
                f"- Crop: `{row['crop_tag']}`",
                f"- Variant: `{row['variant']}`",
                f"- Split: `{row['split_name']}`",
                f"- Default threshold: `{float(row['default_threshold']):.2f}`",
                "",
            ]
        )

    lines.extend(["## QA Regression", "", _frame_to_markdown(qa_metrics), ""])

    test_filter = qa_filtering[qa_filtering["split"].astype(str) == "test"]
    if test_filter.empty:
        test_filter = qa_filtering
    lines.extend(
        [
            "## Threshold Diagnostics",
            "",
            _frame_to_markdown(test_filter),
            "",
        ]
    )

    if not ablation_summary.empty:
        keep_columns = [
            "method_key",
            "fold",
            "family",
            "uses_qa",
            "threshold",
            "fusion_model",
            "mean_iou",
            "p05_iou",
            "bad_image_pct",
            "mean_cells_placed",
            "delta_iou_vs_reference",
            "delta_cells_placed_vs_reference",
        ]
        present = [
            column for column in keep_columns if column in ablation_summary.columns
        ]
        lines.extend(
            [
                "## Ablation Outcomes",
                "",
                _frame_to_markdown(ablation_summary[present]),
                "",
            ]
        )

    if not cross_fold_summary.empty:
        keep_columns = [
            "method_key",
            "family",
            "uses_qa",
            "mean_iou_across_folds",
            "fold_iou_gap",
            "worst_fold_iou",
            "best_fold_iou",
        ]
        present = [
            column for column in keep_columns if column in cross_fold_summary.columns
        ]
        lines.extend(
            [
                "## Cross-Fold Stability",
                "",
                _frame_to_markdown(cross_fold_summary[present]),
                "",
            ]
        )

    triggered = flags[flags["triggered"].fillna(False)] if not flags.empty else flags
    lines.append("## Flags")
    lines.append("")
    if triggered is None or triggered.empty:
        lines.append("- None")
    else:
        for _, row in triggered.iterrows():
            lines.append(f"- `{row['severity']}` `{row['code']}`: {row['detail']}")
    lines.append("")

    default_row = _closest_threshold_row(qa_filtering, default_threshold)
    best_full_pipeline = _best_method(ablation_summary, family="full_pipeline")
    if default_row is not None or best_full_pipeline is not None:
        lines.append("## Interpretation")
        lines.append("")
        if default_row is not None:
            lines.append(
                "- Default threshold diagnostics: "
                f"actual-good `{float(default_row['actual_good_pct']):.2f}%`, "
                f"predicted-good `{float(default_row['predicted_good_pct']):.2f}%`, "
                f"recall `{float(default_row['recall']):.4f}`."
            )
        if best_full_pipeline is not None:
            lines.append(
                "- Best full-pipeline result: "
                f"`{best_full_pipeline['method_key']}` with mean IoU "
                f"`{float(best_full_pipeline['mean_iou']):.4f}`."
            )
        if not cross_fold_summary.empty:
            unstable_method = _fold_instability_row(cross_fold_summary)
            if unstable_method is not None:
                lines.append(
                    "- Largest cross-fold gap: "
                    f"`{unstable_method['method_key']}` with fold-gap "
                    f"`{float(unstable_method['fold_iou_gap']):.4f}`."
                )
        lines.append(
            "- Treat QA gains as a threshold-selection question unless the test-set "
            "regression and threshold metrics are also healthy."
        )
        lines.append("")

    return "\n".join(lines)


generate_qa_ablation_report = generate_ablation_diagnostics_report
write_qa_ablation_report = write_ablation_diagnostics_report


def _best_method(
    df: pd.DataFrame, *, family: str | None = None, uses_qa: bool | None = None
) -> pd.Series | None:
    if df.empty:
        return None
    subset = df
    if family is not None:
        subset = subset[subset["family"] == family]
    if uses_qa is not None:
        subset = subset[subset["uses_qa"] == uses_qa]
    if subset.empty:
        return None
    return subset.sort_values(["mean_iou", "mean_f1"], ascending=[False, False]).iloc[0]


def _threshold_fragility_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    subset = df[df["family"] == "full_pipeline"].dropna(subset=["threshold"])
    if subset.empty:
        return None

    records: list[dict[str, Any]] = []
    for (family, fusion_model), group in subset.groupby(["family", "fusion_model"]):
        if group["threshold"].nunique() < 2:
            continue
        sorted_group = group.sort_values("threshold")
        adjacent_jump = sorted_group["mean_iou"].diff().abs().max()
        records.append(
            {
                "family": family,
                "fusion_model": fusion_model,
                "threshold_range_iou": float(
                    sorted_group["mean_iou"].max() - sorted_group["mean_iou"].min()
                ),
                "max_adjacent_jump_iou": float(adjacent_jump)
                if pd.notna(adjacent_jump)
                else 0.0,
            }
        )

    if not records:
        return None

    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["threshold_range_iou", "max_adjacent_jump_iou"], ascending=[False, False]
        )
        .iloc[0]
    )


def _model_dependence_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    subset = df[df["family"] == "full_pipeline"].dropna(
        subset=["threshold", "fusion_model"]
    )
    if subset.empty:
        return None

    records: list[dict[str, Any]] = []
    for threshold, group in subset.groupby("threshold"):
        if group["fusion_model"].nunique() < 2:
            continue
        records.append(
            {
                "threshold": float(threshold),
                "model_spread_iou": float(
                    group["mean_iou"].max() - group["mean_iou"].min()
                ),
                "best_model": str(
                    group.sort_values("mean_iou", ascending=False).iloc[0][
                        "fusion_model"
                    ]
                ),
                "worst_model": str(
                    group.sort_values("mean_iou", ascending=True).iloc[0][
                        "fusion_model"
                    ]
                ),
            }
        )

    if not records:
        return None

    return (
        pd.DataFrame.from_records(records)
        .sort_values("model_spread_iou", ascending=False)
        .iloc[0]
    )


def _tail_risk_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.sort_values(
        ["bad_image_pct", "p05_iou", "min_iou"], ascending=[False, True, True]
    ).iloc[0]


def _retrained_value_row(
    df: pd.DataFrame, default_threshold: float
) -> pd.Series | None:
    if df.empty:
        return None

    retrained = df[
        (df["family"] == "ensemble_qa_retrained")
        & (df["threshold"].round(2) == round(default_threshold, 2))
    ]
    if retrained.empty:
        return None

    candidates = df[
        (
            (df["family"] == "ensemble_qa")
            & (df["threshold"].round(2) == round(default_threshold, 2))
        )
        | (df["family"] == "ensemble_only")
    ]
    if candidates.empty:
        return None

    row = retrained.iloc[0].copy()
    row["delta_vs_best_baseline"] = float(
        row["mean_iou"] - candidates["mean_iou"].max()
    )
    return row


def _fold_instability_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.sort_values(
        ["fold_iou_gap", "mean_iou_across_folds"], ascending=[False, False]
    ).iloc[0]


def _closest_threshold_row(df: pd.DataFrame, target: float) -> pd.Series | None:
    if df.empty or "threshold" not in df.columns:
        return None
    subset = df[df["split"].astype(str) == "test"]
    if subset.empty:
        subset = df
    distances = (subset["threshold"] - target).abs()
    return subset.loc[distances.idxmin()]


def _first_split_row(df: pd.DataFrame, split_name: str) -> pd.Series | None:
    if df.empty or "split" not in df.columns:
        return None
    subset = df[df["split"].astype(str) == split_name]
    if subset.empty:
        return None
    return subset.iloc[0]


def _append_flag(
    records: list[dict[str, Any]],
    *,
    code: str,
    severity: str,
    triggered: bool,
    detail: str,
) -> None:
    records.append(
        {
            "code": code,
            "severity": severity,
            "triggered": bool(triggered),
            "detail": detail,
        }
    )


def _series_value(series: pd.Series | None, key: str) -> Any:
    if series is None:
        return None
    return series.get(key)


def _frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"

    header = "| " + " | ".join(str(column) for column in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [header, separator]
    for record in df.to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                _format_markdown_value(record[column]) for column in df.columns
            )
            + " |"
        )
    return "\n".join(rows)


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
