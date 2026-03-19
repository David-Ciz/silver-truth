from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
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


def write_hsc_reporting_bundle(output_dir: Path, bundle: dict[str, pd.DataFrame]) -> dict[str, Path]:
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


def _discover_hsc_inventory(
    *,
    paper_runs_root: Path,
    variant: str,
    qa_threshold: float,
    fusion_model: str,
    full_pipeline_model: str,
) -> pd.DataFrame:
    threshold_tag = f"{qa_threshold:.2f}"
    records: list[dict[str, Any]] = []

    for fold in _FOLDS:
        fold_records = [
            {
                "artifact_key": f"{fold}__competitors",
                "method_group": "competitor",
                "method_label": "Competitor baselines",
                "fold": fold,
                "source_type": "competitor_csv",
                "path": paper_runs_root / "baselines" / f"BF-C2DL-HSC_{fold}_competitors.csv",
            },
            {
                "artifact_key": f"{fold}__silver_truth",
                "method_group": "silver_truth",
                "method_label": "SILVER-TRUTH",
                "fold": fold,
                "source_type": "silver_truth_csv",
                "path": paper_runs_root / "baselines" / f"BF-C2DL-HSC_{fold}_silver_truth.csv",
            },
            {
                "artifact_key": f"{fold}__qa_only",
                "method_group": "qa_only",
                "method_label": "qa_only / top-1",
                "fold": fold,
                "source_type": "fullimage_eval_csv",
                "path": paper_runs_root / "ablation" / variant / fold / "qa_only" / "fullimage_eval.csv",
            },
            {
                "artifact_key": f"{fold}__fusion_only__{fusion_model}",
                "method_group": "fusion_only",
                "method_label": f"fusion_only / {fusion_model.upper()}",
                "fold": fold,
                "source_type": "fullimage_eval_csv",
                "path": paper_runs_root
                / "ablation"
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
                    paper_runs_root / "ablation" / variant / fold / "ensemble_only",
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
                    paper_runs_root / "ablation" / variant / fold / f"ensemble_qa_t{threshold_tag}",
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
    for record in inventory.to_dict("records"):
        if not record["exists"]:
            continue
        path = Path(record["path"])
        source_type = record["source_type"]
        fold = record["fold"]
        if source_type in {"competitor_csv", "silver_truth_csv"}:
            frames.append(_load_competitor_metrics(path, fold=fold, source_type=source_type))
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
    return combined.sort_values(["method_group", "method_label", "fold", "image_id"]).reset_index(drop=True)


def _summarize_methods(
    per_image: pd.DataFrame,
    *,
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if per_image.empty:
        return pd.DataFrame()

    fold_summary = (
        per_image.groupby(["method_key", "method_group", "method_label", "fold"], as_index=False)
        .agg(
            n_images=("image_id", "nunique"),
            fold_iou=("iou", "mean"),
            fold_f1=("f1", "mean"),
        )
    )

    overall_rows: list[dict[str, Any]] = []
    for (method_key, method_group, method_label), group in per_image.groupby(
        ["method_key", "method_group", "method_label"]
    ):
        method_fold_summary = fold_summary[fold_summary["method_key"] == method_key]
        iou_ci = bootstrap_mean_ci(
            group["iou"],
            confidence=confidence,
            n_resamples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        f1_ci = bootstrap_mean_ci(
            group["f1"],
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
                int(fold_row["n_images"].iloc[0]) if not fold_row.empty else float("nan")
            )
            row[f"{fold}_iou"] = float(fold_row["fold_iou"].iloc[0]) if not fold_row.empty else float("nan")
            row[f"{fold}_f1"] = float(fold_row["fold_f1"].iloc[0]) if not fold_row.empty else float("nan")
        overall_rows.append(row)

    summary = pd.DataFrame.from_records(overall_rows)
    summary["group_order"] = summary["method_group"].map(_CORE_GROUP_ORDER).fillna(50)
    summary = summary.sort_values(["group_order", "mean_iou", "method_label"], ascending=[True, False, True])
    return summary.drop(columns="group_order").reset_index(drop=True)


def _build_default_comparisons(per_image: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
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
        reference_row = _resolve_method_token(summary, reference_token, best_competitor=best_competitor)
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

    for group in ("silver_truth", "fusion_only", "qa_only", "full_pipeline", "ensemble_only", "ensemble_qa"):
        group_rows = summary[summary["method_group"] == group]
        if not group_rows.empty:
            keep_keys.add(group_rows.iloc[0]["method_key"])

    core = summary[summary["method_key"].isin(keep_keys)].copy()
    if best_competitor is not None:
        core.loc[core["method_key"] == best_competitor["method_key"], "method_group"] = "best_competitor"
        core.loc[core["method_key"] == best_competitor["method_key"], "method_label"] = (
            f"{best_competitor['method_label']} (best competitor)"
        )

    core["group_order"] = core["method_group"].map(_CORE_GROUP_ORDER).fillna(50)
    core = core.sort_values(["group_order", "mean_iou", "method_label"], ascending=[True, False, True])
    return core.drop(columns="group_order").reset_index(drop=True)


def _paired_metric_rows(
    per_image: pd.DataFrame,
    *,
    reference_key: str,
    candidate_key: str,
    metric: str,
) -> dict[str, Any]:
    reference = per_image[per_image["method_key"] == reference_key][["fold", "image_id", metric]].rename(
        columns={metric: "reference_value"}
    )
    candidate = per_image[per_image["method_key"] == candidate_key][["fold", "image_id", metric]].rename(
        columns={metric: "candidate_value"}
    )
    merged = reference.merge(candidate, on=["fold", "image_id"], how="inner")
    stats = paired_wilcoxon_test(merged["reference_value"], merged["candidate_value"])
    stats["paired_images"] = int(len(merged))
    return stats


def _best_competitor_row(summary: pd.DataFrame) -> pd.Series | None:
    competitors = summary[summary["method_group"] == "competitor"]
    if competitors.empty:
        return None
    return competitors.sort_values(["mean_iou", "method_label"], ascending=[False, True]).iloc[0]


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


def _load_competitor_metrics(path: Path, *, fold: str, source_type: str) -> pd.DataFrame:
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

    grouped = (
        filtered.groupby(
            ["method_key", "method_group", "method_label", "image_id"], as_index=False
        )
        .agg(iou=("image_average", "first"), f1=("image_f1_average", "first"))
    )
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
        lambda row: _normalize_fullimage_eval_id(row["campaign_number"], row["original_image_key"]),
        axis=1,
    )
    normalized = df[["image_id", "iou", "f1"]].copy()
    normalized["method_key"] = artifact_key
    normalized["method_group"] = method_group
    normalized["method_label"] = method_label
    normalized["fold"] = fold
    normalized["source_path"] = str(path)
    return normalized[
        ["method_key", "method_group", "method_label", "fold", "image_id", "iou", "f1", "source_path"]
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
        lambda row: _normalize_fullimage_eval_id(row["campaign_number"], row["original_image_key"]),
        axis=1,
    )
    normalized = df[["image_id", "iou", "f1"]].copy()
    normalized["method_key"] = artifact_key
    normalized["method_group"] = method_group
    normalized["method_label"] = method_label
    normalized["fold"] = fold
    normalized["source_path"] = str(path)
    return normalized[
        ["method_key", "method_group", "method_label", "fold", "image_id", "iou", "f1", "source_path"]
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
