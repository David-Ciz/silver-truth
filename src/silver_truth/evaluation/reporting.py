from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from silver_truth.data_processing.segmentation_stats import (
    collect_segmentation_object_stats_from_dataframes,
)
from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)

logger = logging.getLogger(__name__)

_FOLDS = ("fold-1", "fold-2")
_CORE_GROUP_ORDER = {
    "best_competitor": 0,
    "silver_truth": 1,
    "fusion_only": 2,
    "qa_only": 3,
    "full_pipeline": 4,
    "ensemble_only": 5,
    "ensemble_qa": 6,
    "competitor": 99,
}


def bootstrap_mean_ci(
    values: pd.Series | np.ndarray | list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) == 0:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    if len(array) == 1:
        value = float(array[0])
        return {"mean": value, "ci_lower": value, "ci_upper": value}

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(array), size=(n_resamples, len(array)))
    sample_means = array[sample_idx].mean(axis=1)
    alpha = 1.0 - confidence

    return {
        "mean": float(array.mean()),
        "ci_lower": float(np.quantile(sample_means, alpha / 2.0)),
        "ci_upper": float(np.quantile(sample_means, 1.0 - alpha / 2.0)),
    }


def bootstrap_stratified_mean_ci(
    values_by_group: dict[str, pd.Series | np.ndarray | list[float]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    arrays: list[np.ndarray] = []
    for values in values_by_group.values():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if len(array) == 0:
            continue
        arrays.append(array)

    if not arrays:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    observed_group_means = np.asarray([array.mean() for array in arrays], dtype=float)
    observed_mean = float(observed_group_means.mean())

    if n_resamples <= 0 or all(len(array) == 1 for array in arrays):
        return {
            "mean": observed_mean,
            "ci_lower": observed_mean,
            "ci_upper": observed_mean,
        }

    rng = np.random.default_rng(seed)
    resampled_group_means = []
    for array in arrays:
        sample_idx = rng.integers(0, len(array), size=(n_resamples, len(array)))
        resampled_group_means.append(array[sample_idx].mean(axis=1))

    sample_means = np.mean(np.vstack(resampled_group_means), axis=0)
    alpha = 1.0 - confidence

    return {
        "mean": observed_mean,
        "ci_lower": float(np.quantile(sample_means, alpha / 2.0)),
        "ci_upper": float(np.quantile(sample_means, 1.0 - alpha / 2.0)),
    }


def paired_wilcoxon_test(
    reference_values: pd.Series | np.ndarray | list[float],
    candidate_values: pd.Series | np.ndarray | list[float],
) -> dict[str, float | int | str]:
    reference = np.asarray(reference_values, dtype=float)
    candidate = np.asarray(candidate_values, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(candidate)
    reference = reference[valid]
    candidate = candidate[valid]

    if len(reference) == 0:
        return {
            "n_pairs": 0,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "note": "no paired samples",
        }

    deltas = candidate - reference
    if len(reference) < 2:
        return {
            "n_pairs": int(len(reference)),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "note": "fewer than two paired samples",
        }

    if np.allclose(deltas, 0.0):
        return {
            "n_pairs": int(len(reference)),
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "note": "all paired differences are zero",
        }

    statistic, p_value = wilcoxon(candidate, reference, zero_method="wilcox")
    return {
        "n_pairs": int(len(reference)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "note": "",
    }


def generate_hsc_reporting_bundle(
    *,
    paper_runs_root: Path,
    dataset: str = "BF-C2DL-HSC",
    crop_tag: str = "sz64",
    variant: str = "baseline",
    qa_threshold: float = 0.75,
    fusion_model: str = "simple",
    full_pipeline_model: str = "simple",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, pd.DataFrame]:
    inventory = _discover_hsc_inventory(
        paper_runs_root=paper_runs_root,
        dataset=dataset,
        crop_tag=crop_tag,
        variant=variant,
        qa_threshold=qa_threshold,
        fusion_model=fusion_model,
        full_pipeline_model=full_pipeline_model,
    )
    per_image = _load_inventory_records(inventory)
    summary = _summarize_methods(
        per_image,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    comparisons = _build_default_comparisons(per_image, summary)
    core_summary = _build_core_summary(summary)

    return {
        "inventory": inventory,
        "per_image": per_image,
        "summary": summary,
        "core_summary": core_summary,
        "comparisons": comparisons,
    }


def write_hsc_reporting_bundle(
    output_dir: Path, bundle: dict[str, pd.DataFrame]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "inventory": output_dir / "artifact_inventory.csv",
        "per_image": output_dir / "per_image_metrics.csv",
        "summary": output_dir / "method_summary.csv",
        "core_summary": output_dir / "core_method_summary.csv",
        "comparisons": output_dir / "paired_comparisons.csv",
        "markdown": output_dir / "summary_report.md",
    }

    bundle["inventory"].to_csv(paths["inventory"], index=False)
    bundle["per_image"].to_csv(paths["per_image"], index=False)
    bundle["summary"].to_csv(paths["summary"], index=False)
    bundle["core_summary"].to_csv(paths["core_summary"], index=False)
    bundle["comparisons"].to_csv(paths["comparisons"], index=False)
    paths["markdown"].write_text(_render_markdown_report(bundle), encoding="utf-8")

    return paths


def generate_hsc_main_comparison_bundle(
    *,
    comparison_root: Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, pd.DataFrame]:
    per_image = _load_hsc_main_comparison_records(comparison_root)
    summary = _summarize_methods(
        per_image,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    core_summary = _build_core_summary(summary)

    best_competitor = _best_competitor_row(summary)
    comparison_rows: list[dict[str, Any]] = []
    if best_competitor is not None and "ensemble_only" in set(summary["method_key"]):
        for metric in ("iou", "f1"):
            paired = _paired_metric_rows(
                per_image,
                reference_key=str(best_competitor["method_key"]),
                candidate_key="ensemble_only",
                metric=metric,
            )
            paired["metric"] = metric
            paired["reference_method_key"] = str(best_competitor["method_key"])
            paired["reference_method_label"] = str(best_competitor["method_label"])
            paired["candidate_method_key"] = "ensemble_only"
            paired["candidate_method_label"] = "ensemble_only"
            comparison_rows.append(paired)

    comparisons = pd.DataFrame.from_records(comparison_rows)

    limitations = pd.DataFrame.from_records(
        [
            {
                "scope": "main_hsc_comparison",
                "status": "bootstrap_supported",
                "note": (
                    "The main HSC ensemble-only comparison includes 95% "
                    "fold-stratified bootstrap CIs over test images."
                ),
            },
            {
                "scope": "qa_gated_hsc_ablation_rows",
                "status": "descriptive_only",
                "note": (
                    "The broader HSC QA-gated ablation rows remain fold-level "
                    "descriptive results only because the frozen local snapshot "
                    "does not contain the full per-image exports for both folds."
                ),
            },
            {
                "scope": "pooled_paired_test",
                "status": "descriptive_only",
                "note": (
                    "The pooled paired Wilcoxon rows are reported descriptively "
                    "only: they pool image pairs across folds and therefore "
                    "inherit the strong fold-size imbalance (8 vs 49 test images)."
                ),
            },
        ]
    )

    return {
        "per_image": per_image,
        "summary": summary,
        "core_summary": core_summary,
        "comparisons": comparisons,
        "limitations": limitations,
    }


def write_hsc_main_comparison_bundle(
    output_dir: Path, bundle: dict[str, pd.DataFrame]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "per_image": output_dir / "per_image_metrics.csv",
        "summary": output_dir / "method_summary.csv",
        "core_summary": output_dir / "core_method_summary.csv",
        "comparisons": output_dir / "paired_comparisons.csv",
        "limitations": output_dir / "limitations.csv",
        "markdown": output_dir / "summary_report.md",
    }

    bundle["per_image"].to_csv(paths["per_image"], index=False)
    bundle["summary"].to_csv(paths["summary"], index=False)
    bundle["core_summary"].to_csv(paths["core_summary"], index=False)
    bundle["comparisons"].to_csv(paths["comparisons"], index=False)
    bundle["limitations"].to_csv(paths["limitations"], index=False)
    paths["markdown"].write_text(
        _render_main_comparison_markdown_report(bundle), encoding="utf-8"
    )

    return paths


def analyze_overflow_impact(
    *,
    results_path: Path,
    dataset_dataframe_path: Path,
    crop_size: int,
    metric_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Join per-cell result tables with overflow flags derived from GT bbox size."""
    project_root = _find_project_root_for_paths(dataset_dataframe_path)
    results_df = _load_result_table(results_path)
    standardized_results = _standardize_result_rows(
        results_df, project_root=project_root
    )

    dataset_df = load_dataframe_from_parquet_with_metadata(str(dataset_dataframe_path))
    split_lookup = _build_dataset_split_lookup(dataset_df, project_root=project_root)

    stats_df = collect_segmentation_object_stats_from_dataframes(
        [dataset_dataframe_path]
    )
    overflow_lookup = _build_overflow_lookup(stats_df, crop_size=crop_size)

    enriched = standardized_results.merge(
        overflow_lookup,
        on=["gt_image_resolved", "label"],
        how="left",
    ).merge(
        split_lookup,
        on="gt_image_resolved",
        how="left",
        suffixes=("", "_dataset"),
    )

    enriched["overflow_status"] = np.where(
        enriched["is_overflow"].fillna(False), "overflow", "fits"
    )
    enriched["crop_size"] = int(crop_size)

    resolved_metric_columns = _resolve_metric_columns(enriched, metric_columns)

    outputs: dict[str, pd.DataFrame] = {
        "enriched": enriched,
        "overall_summary": _summarize_metric_groups(
            enriched,
            group_cols=["overflow_status"],
            metric_columns=resolved_metric_columns,
        ),
    }

    if "competitor" in enriched.columns:
        outputs["by_competitor_summary"] = _summarize_metric_groups(
            enriched,
            group_cols=["competitor", "overflow_status"],
            metric_columns=resolved_metric_columns,
        )

    if "split" in enriched.columns:
        outputs["by_split_summary"] = _summarize_metric_groups(
            enriched,
            group_cols=["split", "overflow_status"],
            metric_columns=resolved_metric_columns,
        )

    if {"competitor", "split"}.issubset(enriched.columns):
        outputs["by_competitor_split_summary"] = _summarize_metric_groups(
            enriched,
            group_cols=["competitor", "split", "overflow_status"],
            metric_columns=resolved_metric_columns,
        )

    return outputs


def write_overflow_impact_bundle(
    output_dir: Path, bundle: dict[str, pd.DataFrame]
) -> dict[str, Path]:
    """Persist overflow-impact analysis tables to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, frame in bundle.items():
        filename = f"{name}.parquet" if name == "enriched" else f"{name}.csv"
        path = output_dir / filename
        if name == "enriched":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        paths[name] = path

    return paths


def _find_project_root_for_paths(dataset_dataframe_path: Path) -> Path:
    start = dataset_dataframe_path.resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return dataset_dataframe_path.resolve().parent


def _resolve_path_for_reporting(path_value: Any, project_root: Path) -> str | None:
    if path_value is None or pd.isna(path_value):
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def _load_result_table(results_path: Path) -> pd.DataFrame:
    if results_path.suffix == ".csv":
        return pd.read_csv(results_path)
    return pd.read_parquet(results_path)


def _standardize_result_rows(
    results_df: pd.DataFrame, *, project_root: Path
) -> pd.DataFrame:
    df = results_df.copy()
    if "gt_image" not in df.columns and "gt_seg_path" in df.columns:
        df["gt_image"] = df["gt_seg_path"]
    if "label" not in df.columns and "cell_id" in df.columns:
        df["label"] = df["cell_id"]

    required = {"gt_image", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Result table is missing required columns for overflow analysis: "
            + ", ".join(sorted(missing))
        )

    df["gt_image_resolved"] = df["gt_image"].apply(
        lambda value: _resolve_path_for_reporting(value, project_root)
    )
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["gt_image_resolved", "label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def _build_dataset_split_lookup(
    dataset_df: pd.DataFrame, *, project_root: Path
) -> pd.DataFrame:
    available_columns = [
        column
        for column in ["gt_image", "split", "composite_key", "campaign_number"]
        if column in dataset_df.columns
    ]
    if "gt_image" not in available_columns:
        return pd.DataFrame(columns=["gt_image_resolved"])

    lookup = (
        dataset_df[available_columns]
        .dropna(subset=["gt_image"])
        .drop_duplicates(subset=["gt_image"])
    )
    lookup = lookup.copy()
    lookup["gt_image_resolved"] = lookup["gt_image"].apply(
        lambda value: _resolve_path_for_reporting(value, project_root)
    )
    keep_columns = ["gt_image_resolved"] + [
        column
        for column in ["split", "composite_key", "campaign_number"]
        if column in lookup.columns
    ]
    return lookup[keep_columns]


def _build_overflow_lookup(stats_df: pd.DataFrame, *, crop_size: int) -> pd.DataFrame:
    lookup = stats_df.copy()
    lookup["gt_image_resolved"] = lookup["gt_image"].apply(
        lambda value: str(Path(str(value)).resolve())
    )
    lookup["label"] = lookup["label_id"].astype(int)
    lookup["is_overflow"] = (lookup["bbox_height_px"] > crop_size) | (
        lookup["bbox_width_px"] > crop_size
    )
    keep_columns = [
        "gt_image_resolved",
        "label",
        "is_overflow",
        "bbox_height_px",
        "bbox_width_px",
        "bbox_max_dim_px",
        "bbox_aspect_ratio_wh",
        "bbox_elongation_ratio",
    ]
    return lookup[keep_columns]


def _resolve_metric_columns(
    df: pd.DataFrame, metric_columns: list[str] | None
) -> list[str]:
    if metric_columns:
        present = [column for column in metric_columns if column in df.columns]
        if present:
            return present
        raise ValueError(
            "Requested metric columns not found in result table: "
            + ", ".join(metric_columns)
        )

    candidates = ["jaccard_score", "f1_score", "iou", "f1", "predicted_jaccard_index"]
    present = [column for column in candidates if column in df.columns]
    if present:
        return present

    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and column not in {"label", "crop_size"}
    ]
    return numeric_columns


def _summarize_metric_groups(
    df: pd.DataFrame, *, group_cols: list[str], metric_columns: list[str]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_cols, group_key)}
        row["n_rows"] = int(len(group_df))
        row["n_cells"] = int(
            group_df[["gt_image_resolved", "label"]].drop_duplicates().shape[0]
        )
        row["n_gt_images"] = int(group_df["gt_image_resolved"].nunique())

        if "is_overflow" in group_df.columns:
            row["overflow_rate"] = float(group_df["is_overflow"].fillna(False).mean())

        for metric in metric_columns:
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            row[f"{metric}_count"] = int(len(values))
            row[f"{metric}_mean"] = (
                float(values.mean()) if not values.empty else float("nan")
            )
            row[f"{metric}_median"] = (
                float(values.median()) if not values.empty else float("nan")
            )
            row[f"{metric}_p05"] = (
                float(values.quantile(0.05)) if not values.empty else float("nan")
            )
            row[f"{metric}_p95"] = (
                float(values.quantile(0.95)) if not values.empty else float("nan")
            )

        records.append(row)

    return pd.DataFrame.from_records(records)


def _discover_hsc_inventory(
    *,
    paper_runs_root: Path,
    dataset: str,
    crop_tag: str,
    variant: str,
    qa_threshold: float,
    fusion_model: str,
    full_pipeline_model: str,
) -> pd.DataFrame:
    threshold_tag = f"{qa_threshold:.2f}"
    records: list[dict[str, Any]] = []

    for fold in _FOLDS:
        fold_records: list[dict[str, Any]] = [
            {
                "artifact_key": f"{fold}__competitors",
                "method_group": "competitor",
                "method_label": "Competitor baselines",
                "fold": fold,
                "source_type": "competitor_csv",
                "path": paper_runs_root
                / "baselines"
                / f"{dataset}_{crop_tag}_{variant}_{fold}_competitors.csv",
            },
            {
                "artifact_key": f"{fold}__silver_truth",
                "method_group": "silver_truth",
                "method_label": "SILVER-TRUTH",
                "fold": fold,
                "source_type": "silver_truth_csv",
                "path": paper_runs_root
                / "baselines"
                / f"{dataset}_{crop_tag}_{variant}_{fold}_silver_truth.csv",
            },
            {
                "artifact_key": f"{fold}__qa_only",
                "method_group": "qa_only",
                "method_label": "qa_only / top-1",
                "fold": fold,
                "source_type": "fullimage_eval_csv",
                "path": paper_runs_root
                / "ablation"
                / dataset
                / crop_tag
                / variant
                / fold
                / "qa_only"
                / "fullimage_eval.csv",
            },
            {
                "artifact_key": f"{fold}__fusion_only__{fusion_model}",
                "method_group": "fusion_only",
                "method_label": f"fusion_only / {fusion_model.upper()}",
                "fold": fold,
                "source_type": "fullimage_eval_csv",
                "path": paper_runs_root
                / "ablation"
                / dataset
                / crop_tag
                / variant
                / fold
                / "fusion_only"
                / fusion_model
                / "fullimage_eval.csv",
            },
            {
                "artifact_key": f"{fold}__full_pipeline_t{threshold_tag}__{full_pipeline_model}",
                "method_group": "full_pipeline",
                "method_label": f"full_pipeline / {full_pipeline_model.upper()} @ t={threshold_tag}",
                "fold": fold,
                "source_type": "fullimage_eval_csv",
                "path": paper_runs_root
                / "ablation"
                / dataset
                / crop_tag
                / variant
                / fold
                / f"full_pipeline_t{threshold_tag}"
                / full_pipeline_model
                / "fullimage_eval.csv",
            },
            {
                "artifact_key": f"{fold}__ensemble_only",
                "method_group": "ensemble_only",
                "method_label": "ensemble_only",
                "fold": fold,
                "source_type": "ensemble_parquet",
                "path": _first_matching_path(
                    paper_runs_root
                    / "ablation"
                    / dataset
                    / crop_tag
                    / variant
                    / fold
                    / "ensemble_only",
                    "*_set-test.parquet",
                ),
            },
            {
                "artifact_key": f"{fold}__ensemble_qa_t{threshold_tag}",
                "method_group": "ensemble_qa",
                "method_label": f"ensemble_qa @ t={threshold_tag}",
                "fold": fold,
                "source_type": "ensemble_parquet",
                "path": _first_matching_path(
                    paper_runs_root
                    / "ablation"
                    / dataset
                    / crop_tag
                    / variant
                    / fold
                    / f"ensemble_qa_t{threshold_tag}",
                    "*_set-test.parquet",
                ),
            },
        ]

        for record in fold_records:
            resolved_path = record["path"]
            exists = bool(resolved_path and Path(resolved_path).exists())
            record["path"] = str(resolved_path) if resolved_path else ""
            record["exists"] = exists
            records.append(record)

    return pd.DataFrame.from_records(records)


def _load_inventory_records(inventory: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_record in inventory.to_dict("records"):
        record = cast(dict[str, Any], raw_record)
        if not record["exists"]:
            continue
        path = Path(record["path"])
        source_type = record["source_type"]
        fold = record["fold"]
        if source_type in {"competitor_csv", "silver_truth_csv"}:
            frames.append(
                _load_competitor_metrics(path, fold=fold, source_type=source_type)
            )
        elif source_type == "fullimage_eval_csv":
            frames.append(
                _load_fullimage_eval(
                    path,
                    fold=fold,
                    method_group=record["method_group"],
                    method_label=record["method_label"],
                    artifact_key=record["artifact_key"],
                )
            )
        elif source_type == "ensemble_parquet":
            frames.append(
                _load_ensemble_eval(
                    path,
                    fold=fold,
                    method_group=record["method_group"],
                    method_label=record["method_label"],
                    artifact_key=record["artifact_key"],
                )
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "method_key",
                "method_group",
                "method_label",
                "fold",
                "image_id",
                "iou",
                "f1",
                "source_path",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["method_key", "fold", "image_id"])
    return combined.sort_values(
        ["method_group", "method_label", "fold", "image_id"]
    ).reset_index(drop=True)


def _summarize_methods(
    per_image: pd.DataFrame,
    *,
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if per_image.empty:
        return pd.DataFrame()

    fold_summary = per_image.groupby(
        ["method_key", "method_group", "method_label", "fold"], as_index=False
    ).agg(
        n_images=("image_id", "nunique"),
        fold_iou=("iou", "mean"),
        fold_f1=("f1", "mean"),
    )

    overall_rows: list[dict[str, Any]] = []
    for (method_key, method_group, method_label), group in per_image.groupby(
        ["method_key", "method_group", "method_label"]
    ):
        method_fold_summary = fold_summary[fold_summary["method_key"] == method_key]
        iou_ci = bootstrap_stratified_mean_ci(
            {
                str(fold): fold_group["iou"]
                for fold, fold_group in group.groupby("fold", sort=False)
            },
            confidence=confidence,
            n_resamples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        f1_ci = bootstrap_stratified_mean_ci(
            {
                str(fold): fold_group["f1"]
                for fold, fold_group in group.groupby("fold", sort=False)
            },
            confidence=confidence,
            n_resamples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        row: dict[str, Any] = {
            "method_key": method_key,
            "method_group": method_group,
            "method_label": method_label,
            "n_images_total": int(group["image_id"].nunique()),
            "mean_iou": float(method_fold_summary["fold_iou"].mean()),
            "mean_f1": float(method_fold_summary["fold_f1"].mean()),
            "bootstrap_iou_ci_lower": iou_ci["ci_lower"],
            "bootstrap_iou_ci_upper": iou_ci["ci_upper"],
            "bootstrap_f1_ci_lower": f1_ci["ci_lower"],
            "bootstrap_f1_ci_upper": f1_ci["ci_upper"],
        }
        for fold in _FOLDS:
            fold_row = method_fold_summary[method_fold_summary["fold"] == fold]
            row[f"{fold}_n_images"] = (
                int(fold_row["n_images"].iloc[0])
                if not fold_row.empty
                else float("nan")
            )
            row[f"{fold}_iou"] = (
                float(fold_row["fold_iou"].iloc[0])
                if not fold_row.empty
                else float("nan")
            )
            row[f"{fold}_f1"] = (
                float(fold_row["fold_f1"].iloc[0])
                if not fold_row.empty
                else float("nan")
            )
        overall_rows.append(row)

    summary = pd.DataFrame.from_records(overall_rows)
    summary["group_order"] = summary["method_group"].map(_CORE_GROUP_ORDER).fillna(50)
    summary = summary.sort_values(
        ["group_order", "mean_iou", "method_label"], ascending=[True, False, True]
    )
    return summary.drop(columns="group_order").reset_index(drop=True)


def _build_default_comparisons(
    per_image: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    if per_image.empty or summary.empty:
        return pd.DataFrame()

    best_competitor = _best_competitor_row(summary)
    comparison_specs = [
        ("full_pipeline", "fusion_only"),
        ("full_pipeline", "__best_competitor__"),
        ("ensemble_only", "__best_competitor__"),
    ]

    records: list[dict[str, Any]] = []
    for candidate_token, reference_token in comparison_specs:
        candidate_row = _resolve_method_token(summary, candidate_token)
        reference_row = _resolve_method_token(
            summary, reference_token, best_competitor=best_competitor
        )
        if candidate_row is None or reference_row is None:
            continue

        for metric in ("iou", "f1"):
            paired = _paired_metric_rows(
                per_image,
                reference_key=reference_row["method_key"],
                candidate_key=candidate_row["method_key"],
                metric=metric,
            )
            paired["metric"] = metric
            paired["reference_method_key"] = reference_row["method_key"]
            paired["reference_method_label"] = reference_row["method_label"]
            paired["candidate_method_key"] = candidate_row["method_key"]
            paired["candidate_method_label"] = candidate_row["method_label"]
            records.append(paired)

    return pd.DataFrame.from_records(records)


def _build_core_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    best_competitor = _best_competitor_row(summary)
    keep_keys = set()
    if best_competitor is not None:
        keep_keys.add(best_competitor["method_key"])

    for group in (
        "silver_truth",
        "fusion_only",
        "qa_only",
        "full_pipeline",
        "ensemble_only",
        "ensemble_qa",
    ):
        group_rows = summary[summary["method_group"] == group]
        if not group_rows.empty:
            keep_keys.add(group_rows.iloc[0]["method_key"])

    core = summary[summary["method_key"].isin(keep_keys)].copy()
    if best_competitor is not None:
        core.loc[
            core["method_key"] == best_competitor["method_key"], "method_group"
        ] = "best_competitor"
        core.loc[
            core["method_key"] == best_competitor["method_key"], "method_label"
        ] = f"{best_competitor['method_label']} (best competitor)"

    core["group_order"] = core["method_group"].map(_CORE_GROUP_ORDER).fillna(50)
    core = core.sort_values(
        ["group_order", "mean_iou", "method_label"], ascending=[True, False, True]
    )
    return core.drop(columns="group_order").reset_index(drop=True)


def _paired_metric_rows(
    per_image: pd.DataFrame,
    *,
    reference_key: str,
    candidate_key: str,
    metric: str,
) -> dict[str, Any]:
    reference = per_image[per_image["method_key"] == reference_key][
        ["fold", "image_id", metric]
    ].rename(columns={metric: "reference_value"})
    candidate = per_image[per_image["method_key"] == candidate_key][
        ["fold", "image_id", metric]
    ].rename(columns={metric: "candidate_value"})
    merged = reference.merge(candidate, on=["fold", "image_id"], how="inner")
    stats = paired_wilcoxon_test(merged["reference_value"], merged["candidate_value"])
    stats["paired_images"] = int(len(merged))
    return stats


def _best_competitor_row(summary: pd.DataFrame) -> pd.Series | None:
    competitors = summary[summary["method_group"] == "competitor"]
    if competitors.empty:
        return None
    return competitors.sort_values(
        ["mean_iou", "method_label"], ascending=[False, True]
    ).iloc[0]


def _resolve_method_token(
    summary: pd.DataFrame,
    token: str,
    *,
    best_competitor: pd.Series | None = None,
) -> pd.Series | None:
    if token == "__best_competitor__":
        return best_competitor

    exact = summary[summary["method_key"] == token]
    if not exact.empty:
        return exact.iloc[0]

    group_match = summary[summary["method_group"] == token]
    if not group_match.empty:
        return group_match.iloc[0]

    return None


def _load_hsc_main_comparison_records(comparison_root: Path) -> pd.DataFrame:
    frames = []
    for fold in _FOLDS:
        fold_root = comparison_root / fold
        frames.append(
            _load_competitor_metrics(
                fold_root / "competitors.csv",
                fold=fold,
                source_type="competitor_csv",
            )
        )
        frames.append(
            _load_competitor_metrics(
                fold_root / "silver_truth.csv",
                fold=fold,
                source_type="silver_truth_csv",
            )
        )
        frames.append(
            _load_ensemble_eval(
                fold_root / "checkpoints" / "C1_ds1-42-7015_QA--_M2--_set-test.parquet",
                fold=fold,
                method_group="ensemble_only",
                method_label="ensemble_only",
                artifact_key="ensemble_only",
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["method_key", "fold", "image_id"])
    return combined.sort_values(
        ["method_group", "method_label", "fold", "image_id"]
    ).reset_index(drop=True)


def _load_competitor_metrics(
    path: Path, *, fold: str, source_type: str
) -> pd.DataFrame:
    df = pd.read_csv(path)
    image_col = "image_key"
    required = {"competitor", image_col, "image_average", "image_f1_average"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    test_prefix = "02_" if fold == "fold-1" else "01_"
    filtered = df[df[image_col].astype(str).str.startswith(test_prefix)].copy()
    filtered["image_id"] = filtered[image_col].map(_normalize_competitor_image_id)

    if source_type == "silver_truth_csv":
        filtered["method_key"] = "silver_truth"
        filtered["method_group"] = "silver_truth"
        filtered["method_label"] = "SILVER-TRUTH"
    else:
        filtered["method_key"] = filtered["competitor"].map(
            lambda value: f"competitor__{_slugify(str(value))}"
        )
        filtered["method_group"] = "competitor"
        filtered["method_label"] = filtered["competitor"].astype(str)

    grouped = filtered.groupby(
        ["method_key", "method_group", "method_label", "image_id"], as_index=False
    ).agg(iou=("image_average", "first"), f1=("image_f1_average", "first"))
    grouped["fold"] = fold
    grouped["source_path"] = str(path)
    return grouped


def _load_fullimage_eval(
    path: Path,
    *,
    fold: str,
    method_group: str,
    method_label: str,
    artifact_key: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"campaign_number", "original_image_key", "iou", "f1"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    if "split" in df.columns:
        df = df[df["split"].astype(str) == "test"].copy()

    df["image_id"] = df.apply(
        lambda row: _normalize_fullimage_eval_id(
            row["campaign_number"], row["original_image_key"]
        ),
        axis=1,
    )
    normalized = df[["image_id", "iou", "f1"]].copy()
    normalized["method_key"] = artifact_key
    normalized["method_group"] = method_group
    normalized["method_label"] = method_label
    normalized["fold"] = fold
    normalized["source_path"] = str(path)
    return normalized[
        [
            "method_key",
            "method_group",
            "method_label",
            "fold",
            "image_id",
            "iou",
            "f1",
            "source_path",
        ]
    ]


def _load_ensemble_eval(
    path: Path,
    *,
    fold: str,
    method_group: str,
    method_label: str,
    artifact_key: str,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"campaign_number", "original_image_key", "iou", "f1"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    if "split" in df.columns:
        df = df[df["split"].astype(str) == "test"].copy()

    df["image_id"] = df.apply(
        lambda row: _normalize_fullimage_eval_id(
            row["campaign_number"], row["original_image_key"]
        ),
        axis=1,
    )
    normalized = df[["image_id", "iou", "f1"]].copy()
    normalized["method_key"] = artifact_key
    normalized["method_group"] = method_group
    normalized["method_label"] = method_label
    normalized["fold"] = fold
    normalized["source_path"] = str(path)
    return normalized[
        [
            "method_key",
            "method_group",
            "method_label",
            "fold",
            "image_id",
            "iou",
            "f1",
            "source_path",
        ]
    ]


def _first_matching_path(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _normalize_competitor_image_id(value: str) -> str:
    stem = Path(str(value)).stem
    parts = stem.split("_", maxsplit=1)
    if len(parts) != 2:
        return stem
    seq, frame = parts
    if not frame.startswith("t"):
        frame = f"t{frame}"
    return f"{seq}_{frame}"


def _normalize_fullimage_eval_id(campaign_number: Any, original_image_key: Any) -> str:
    seq = _normalize_sequence_token(campaign_number)
    key = str(original_image_key)
    if not key.startswith("t"):
        key = f"t{key}"
    return f"{seq}_{key}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _normalize_sequence_token(value: Any) -> str:
    text = str(value)
    digits = re.findall(r"\d+", text)
    if digits:
        return digits[-1].zfill(2)
    return text


def _render_markdown_report(bundle: dict[str, pd.DataFrame]) -> str:
    inventory = bundle["inventory"]
    core_summary = bundle["core_summary"]
    comparisons = bundle["comparisons"]

    missing = inventory[~inventory["exists"]][["fold", "method_group", "path"]]

    lines = [
        "# HSC Reporting Summary",
        "",
        "## Core Methods",
        "",
    ]
    if core_summary.empty:
        lines.append("No reportable methods were loaded.")
    else:
        lines.append(core_summary.to_markdown(index=False, floatfmt=".4f"))

    lines.extend(["", "## Paired Comparisons", ""])
    if comparisons.empty:
        lines.append("No paired comparisons were available.")
    else:
        lines.append(comparisons.to_markdown(index=False, floatfmt=".4f"))

    lines.extend(["", "## Missing Artifacts", ""])
    if missing.empty:
        lines.append("No missing artifacts in the requested baseline report.")
    else:
        lines.append(missing.to_markdown(index=False))

    return "\n".join(lines) + "\n"


def _render_main_comparison_markdown_report(bundle: dict[str, pd.DataFrame]) -> str:
    core_summary = bundle["core_summary"]
    comparisons = bundle["comparisons"]
    limitations = bundle["limitations"]

    lines = [
        "# HSC Main Comparison Statistical Note",
        "",
        "## Main Comparison",
        "",
    ]

    if core_summary.empty:
        lines.append("No reportable methods were loaded.")
    else:
        lines.append(core_summary.to_markdown(index=False, floatfmt=".4f"))

    lines.extend(["", "## Paired Comparison", ""])
    if comparisons.empty:
        lines.append("No paired comparisons were available.")
    else:
        lines.append(comparisons.to_markdown(index=False, floatfmt=".4f"))

    lines.extend(["", "## Scope Note", ""])
    lines.append(limitations.to_markdown(index=False))

    return "\n".join(lines) + "\n"
