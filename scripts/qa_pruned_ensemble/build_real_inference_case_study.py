#!/usr/bin/env python
"""Build a real QA-pruned ENSEMBLE inference case-study document."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-silver-truth")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/silver-truth-cache")

import joblib
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]

CASE = {
    "dataset": "Fluo-C2DL-MSC",
    "crop_tag": "sz1024",
    "variant": "msc_crop1024",
    "fold": "fold-1",
    "campaign_number": "02",
    "campaign_number_eval": 2,
    "original_image_key": "t011",
    "label": "1",
}

RUN_ROOT = (
    ROOT
    / "hpc_results/soft_hybrid_probe/qa_pruned_ensemble/learned_candidate_set_weight/"
    / "deep_sets_ltv_clipped_score_boundary_floor075_thr050_trainonly_ltv_boundary_floor075_e080"
)
CANDIDATE_PARQUET = (
    RUN_ROOT
    / "calibrated_features/raw_cluster_softmax_t1p000/Fluo-C2DL-MSC/sz1024/"
    / "msc_crop1024/fold-1/source_balanced_topk/"
    / "clipped_score_boundary_floor075_thr050/prepared_soft_gate_candidates.parquet"
)
FULL_CANDIDATE_PARQUET = (
    ROOT
    / "hpc_results/soft_hybrid_probe/"
    / "all_available_square_rect_full_image_local_support_pairwise_source_trainonly/"
    / "features/Fluo-C2DL-MSC/sz1024/msc_crop1024/"
    / "fold-1_paper_ready_pairwise_ranked.parquet"
)
EVAL_DIR = (
    RUN_ROOT
    / "object_vote/Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-1/"
    / "source_balanced_topk/clipped_score_boundary_floor075_thr050/"
    / "raw_cluster_softmax_t1p000/thr048"
)
MODEL_PAYLOAD = RUN_ROOT / "models/Fluo-C2DL-MSC_fold-1_leave_target_video_out.joblib"
FINAL_ROWS = (
    ROOT
    / "hpc_results/soft_hybrid_probe/qa_pruned_ensemble/final_paper/"
    / "strict_qa_ensemble_final_overall.csv"
)

OUT_DIR = ROOT / "docs/qa_pruned_ensemble_real_case_2026_06_03_assets"
TEX_PATH = ROOT / "docs/qa_pruned_ensemble_real_inference_case_study_2026_06_03.tex"

WEIGHT_COLUMN = "candidate_set_weight_raw_cluster_softmax_t1p000"
RAW_WEIGHT_COLUMN = "candidate_set_weight_raw_score"
THRESHOLD = 0.48
BOUNDARY_FLOOR = 0.75
BOUNDARY_ITERATIONS = 1
WEIGHT_TEMPERATURE = 1.0
TARGET_TEMPERATURE = 0.08


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    retained = _load_case_rows(CANDIDATE_PARQUET)
    full = _load_case_rows(FULL_CANDIDATE_PARQUET)
    eval_row = _load_eval_row()
    payload = joblib.load(MODEL_PAYLOAD)

    retained = retained.sort_values("pairwise_cluster_rank", ascending=False).copy()
    full = full.sort_values("pairwise_cluster_rank", ascending=False).copy()
    retained_sources = set(retained["candidate_source"].astype(str))
    full["retained_for_ensemble"] = full["candidate_source"].astype(str).isin(
        retained_sources
    )
    full["removed_before_vote"] = ~full["retained_for_ensemble"]

    masks, raw_image, gt_mask = _load_candidate_masks(retained)
    fused_mask = _read_mask(Path(eval_row["fused_path"]))
    eval_gt_mask = _read_mask(Path(eval_row["gt_path"]))
    if gt_mask is None:
        gt_mask = eval_gt_mask

    vote, weighted_terms, support_values = _boundary_aware_vote(retained, masks)
    weight_sum = float(retained[WEIGHT_COLUMN].sum())
    fused_recomputed = (vote / weight_sum >= THRESHOLD).astype(np.uint8)
    if fused_recomputed.shape == fused_mask.shape and not np.array_equal(
        fused_recomputed, fused_mask
    ):
        raise RuntimeError("Recomputed fused mask does not match stored fused mask.")

    full_table = _full_table(full, retained_sources)
    retained_table = _retained_table(retained)
    pixel_table = _pixel_table(retained, masks, gt_mask, fused_mask, vote)
    metadata = _metadata_table(payload, eval_row, retained, full)

    full_table.to_csv(OUT_DIR / "case_full_candidate_table.csv", index=False)
    retained_table.to_csv(OUT_DIR / "case_retained_candidate_table.csv", index=False)
    pixel_table.to_csv(OUT_DIR / "case_pixel_accounting.csv", index=False)
    metadata.to_csv(OUT_DIR / "case_model_metadata.csv", index=False)

    _plot_pruning_panel(full, retained_sources, OUT_DIR / "candidate_pruning_panel.png")
    _plot_vote_panel(
        retained,
        masks,
        raw_image,
        gt_mask,
        fused_mask,
        vote / weight_sum,
        OUT_DIR / "retained_vote_panel.png",
    )
    _plot_weight_bar(retained, OUT_DIR / "candidate_weight_bar.png")

    tex = _build_tex(
        retained=retained,
        full=full,
        retained_table=retained_table,
        full_table=full_table,
        pixel_table=pixel_table,
        metadata=metadata,
        eval_row=eval_row,
        payload=payload,
    )
    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"Wrote {TEX_PATH.relative_to(ROOT)}")
    print(f"Wrote assets in {OUT_DIR.relative_to(ROOT)}")


def _normalize_campaign(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(-999).astype(int).astype(str)


def _load_case_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["_campaign_norm"] = _normalize_campaign(frame["campaign_number"])
    mask = (
        frame["_campaign_norm"].eq(str(CASE["campaign_number_eval"]))
        & frame["original_image_key"].astype(str).eq(CASE["original_image_key"])
        & frame["label"].astype(str).eq(CASE["label"])
    )
    result = frame.loc[mask].copy()
    if result.empty:
        raise RuntimeError(f"Case rows not found in {path}")
    if "stacked_path" not in result.columns and "source_path" in result.columns:
        result = _attach_stacked_paths(result)
    return result


def _attach_stacked_paths(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for source_path, rows in frame.groupby("source_path", dropna=False):
        source = pd.read_parquet(_resolve_path(source_path))
        key_column = "cell_id" if "cell_id" in source.columns else "candidate_id"
        source = source[[key_column, "stacked_path"]].drop_duplicates(key_column)
        merged = rows.merge(
            source,
            left_on="candidate_id",
            right_on=key_column,
            how="left",
            suffixes=("", "_source"),
        )
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True)
    if result["stacked_path"].isna().any():
        missing = result.loc[result["stacked_path"].isna(), "candidate_id"].tolist()
        raise RuntimeError(f"Could not recover stacked_path for {missing}")
    return result


def _load_eval_row() -> pd.Series:
    frame = pd.read_csv(EVAL_DIR / "weighted_object_vote_cell_eval.csv")
    mask = (
        pd.to_numeric(frame["campaign_number"], errors="coerce").eq(
            CASE["campaign_number_eval"]
        )
        & frame["original_image_key"].astype(str).eq(CASE["original_image_key"])
        & frame["label"].astype(str).eq(CASE["label"])
    )
    rows = frame.loc[mask].copy()
    if len(rows) != 1:
        raise RuntimeError(f"Expected one eval row, found {len(rows)}")
    return rows.iloc[0]


def _resolve_path(value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    return path


def _read_stack(path: Path) -> np.ndarray:
    stack = tifffile.imread(path)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise RuntimeError(f"Unsupported stack shape {stack.shape}: {path}")
    return stack


def _read_mask(path: Path) -> np.ndarray:
    mask = tifffile.imread(path)
    if mask.ndim == 3:
        mask = mask[0]
    return (mask > 0).astype(np.uint8)


def _load_candidate_masks(
    retained: pd.DataFrame,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray | None]:
    masks: list[np.ndarray] = []
    raw_image: np.ndarray | None = None
    gt_mask: np.ndarray | None = None
    for _, row in retained.iterrows():
        stack = _read_stack(_resolve_path(row["stacked_path"]))
        if raw_image is None:
            raw_image = stack[0].astype(float)
        masks.append((stack[1] > 0).astype(np.float32))
        if gt_mask is None and stack.shape[0] >= 3:
            gt_mask = (stack[2] > 0).astype(np.uint8)
    if raw_image is None:
        raise RuntimeError("No candidate masks loaded.")
    return masks, raw_image, gt_mask


def _binary_erosion_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    eroded = np.ones(mask.shape, dtype=bool)
    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    return eroded


def _mask_boundary(mask: np.ndarray, iterations: int) -> np.ndarray:
    eroded = mask.astype(bool)
    for _ in range(iterations):
        eroded = _binary_erosion_3x3(eroded)
    return mask.astype(bool) & ~eroded


def _boundary_aware_vote(
    retained: pd.DataFrame, masks: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    weights = retained[WEIGHT_COLUMN].astype(float).to_numpy()
    support_values = (
        retained["local_image_support_score"]
        .astype(float)
        .clip(lower=BOUNDARY_FLOOR, upper=1.0)
        .to_numpy()
    )
    vote = np.zeros_like(masks[0], dtype=np.float32)
    terms: list[np.ndarray] = []
    for mask, weight, support in zip(masks, weights, support_values, strict=True):
        boundary = _mask_boundary(mask > 0, BOUNDARY_ITERATIONS)
        pixel_weight = np.where(boundary, float(weight * support), float(weight))
        term = mask * pixel_weight.astype(np.float32)
        terms.append(term)
        vote += term
    return vote, terms, support_values


def _full_table(full: pd.DataFrame, retained_sources: set[str]) -> pd.DataFrame:
    columns = [
        "candidate_source",
        "true_iou",
        "true_f1",
        "qa_score",
        "pairwise_cluster_score",
        "pairwise_cluster_rank",
        "fusion_support",
        "fusion_support_rank_in_image",
        "local_image_support_score",
    ]
    table = full.loc[:, [c for c in columns if c in full.columns]].copy()
    table.insert(0, "status", "removed")
    table.loc[
        table["candidate_source"].astype(str).isin(retained_sources), "status"
    ] = "retained"
    table.insert(0, "rank_by_pairwise", np.arange(1, len(table) + 1))
    return table


def _retained_table(retained: pd.DataFrame) -> pd.DataFrame:
    table = retained[
        [
            "candidate_source",
            "true_iou",
            "true_f1",
            "qa_score",
            "pairwise_cluster_score",
            "pairwise_cluster_rank",
            "local_image_support_score",
            RAW_WEIGHT_COLUMN,
            WEIGHT_COLUMN,
            "_qa_soft_gate_weight",
        ]
    ].copy()
    max_raw = float(table[RAW_WEIGHT_COLUMN].max())
    max_iou = float(table["true_iou"].max())
    table["training_target_for_this_cluster"] = np.exp(
        ((table["true_iou"] - max_iou) / TARGET_TEMPERATURE).clip(lower=-50.0, upper=0.0)
    )
    table["softmax_formula_value"] = np.exp(
        ((table[RAW_WEIGHT_COLUMN] - max_raw) / WEIGHT_TEMPERATURE).clip(
            lower=-50.0, upper=0.0
        )
    )
    table["normalized_vote_weight"] = table[WEIGHT_COLUMN] / table[WEIGHT_COLUMN].sum()
    table["effective_boundary_support"] = table["local_image_support_score"].clip(
        lower=BOUNDARY_FLOOR, upper=1.0
    )
    return table


def _metric(mask: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, float, float]:
    mask_bool = mask.astype(bool)
    gt_bool = gt.astype(bool)
    inter = int(np.logical_and(mask_bool, gt_bool).sum())
    union = int(np.logical_or(mask_bool, gt_bool).sum())
    pred_area = int(mask_bool.sum())
    gt_area = int(gt_bool.sum())
    iou = inter / union if union else math.nan
    f1 = (2 * inter) / (pred_area + gt_area) if pred_area + gt_area else math.nan
    return pred_area, inter, union, iou, f1


def _pixel_table(
    retained: pd.DataFrame,
    masks: list[np.ndarray],
    gt_mask: np.ndarray,
    fused_mask: np.ndarray,
    vote: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for (_, row), mask in zip(retained.iterrows(), masks, strict=True):
        boundary = _mask_boundary(mask > 0, BOUNDARY_ITERATIONS)
        pred_area, inter, union, iou, f1 = _metric(mask, gt_mask)
        rows.append(
            {
                "object": str(row["candidate_source"]),
                "mask_area_px": pred_area,
                "gt_area_px": int(gt_mask.astype(bool).sum()),
                "intersection_px": inter,
                "union_px": union,
                "iou": iou,
                "f1": f1,
                "boundary_px": int(boundary.sum()),
                "interior_px": int((mask.astype(bool) & ~boundary).sum()),
                "deep_sets_weight": float(row[WEIGHT_COLUMN]),
                "boundary_support": float(
                    max(BOUNDARY_FLOOR, min(1.0, row["local_image_support_score"]))
                ),
            }
        )
    pred_area, inter, union, iou, f1 = _metric(fused_mask, gt_mask)
    rows.append(
        {
            "object": "QA-pruned ENSEMBLE fused output",
            "mask_area_px": pred_area,
            "gt_area_px": int(gt_mask.astype(bool).sum()),
            "intersection_px": inter,
            "union_px": union,
            "iou": iou,
            "f1": f1,
            "boundary_px": int(_mask_boundary(fused_mask > 0, BOUNDARY_ITERATIONS).sum()),
            "interior_px": int(
                (fused_mask.astype(bool) & ~_mask_boundary(fused_mask > 0, 1)).sum()
            ),
            "deep_sets_weight": math.nan,
            "boundary_support": math.nan,
        }
    )
    rows.append(
        {
            "object": "vote map before threshold",
            "mask_area_px": int((vote > 0).sum()),
            "gt_area_px": int(gt_mask.astype(bool).sum()),
            "intersection_px": math.nan,
            "union_px": math.nan,
            "iou": math.nan,
            "f1": math.nan,
            "boundary_px": math.nan,
            "interior_px": math.nan,
            "deep_sets_weight": float(retained[WEIGHT_COLUMN].sum()),
            "boundary_support": math.nan,
        }
    )
    return pd.DataFrame.from_records(rows)


def _metadata_table(
    payload: dict, eval_row: pd.Series, retained: pd.DataFrame, full: pd.DataFrame
) -> pd.DataFrame:
    config = payload["config"]
    categories = payload["preprocess"]["categories"]
    summary = pd.read_csv(FINAL_ROWS)
    best = summary[
        summary["selection_rule"].eq("validation_mean_minus_2p000std_selected")
    ].iloc[0]
    return pd.DataFrame(
        [
            ("selected_strict_mean_iou", best["mean_iou"]),
            ("selected_strict_mean_f1", best["mean_f1"]),
            ("case_dataset", CASE["dataset"]),
            ("case_fold", CASE["fold"]),
            ("case_split", "test"),
            ("case_campaign", CASE["campaign_number"]),
            ("case_frame", CASE["original_image_key"]),
            ("case_label", CASE["label"]),
            ("upstream_candidates", len(full)),
            ("retained_candidates", len(retained)),
            ("fused_iou", eval_row["jaccard"]),
            ("fused_f1", eval_row["f1"]),
            ("vote_threshold", THRESHOLD),
            ("boundary_support_column", "local_image_support_score"),
            ("boundary_floor", BOUNDARY_FLOOR),
            ("boundary_iterations", BOUNDARY_ITERATIONS),
            ("weights_column", WEIGHT_COLUMN),
            ("weight_transform", "raw_cluster_softmax_t1p000"),
            ("deep_sets_input_dim", payload["input_dim"]),
            ("deep_sets_numeric_features", len(payload["feature_columns"])),
            ("deep_sets_categorical_columns", len(payload["categorical_columns"])),
            (
                "deep_sets_one_hot_features",
                sum(len(values) for values in categories.values()),
            ),
            ("deep_sets_hidden_dim", config["deep_sets_hidden_dim"]),
            ("deep_sets_epochs", config["deep_sets_epochs"]),
            ("deep_sets_learning_rate", config["deep_sets_learning_rate"]),
            ("deep_sets_weight_decay", config["deep_sets_weight_decay"]),
            ("deep_sets_train_rows", payload["train_rows"]),
            ("deep_sets_train_clusters", payload["train_clusters"]),
            ("deep_sets_final_loss", payload["train_losses"][-1]),
        ],
        columns=["name", "value"],
    )


def _crop_bounds(masks: Iterable[np.ndarray], pad: int = 20) -> tuple[slice, slice]:
    total = None
    for mask in masks:
        current = mask.astype(bool)
        total = current if total is None else (total | current)
    if total is None or not total.any():
        return slice(None), slice(None)
    ys, xs = np.where(total)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(total.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(total.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def _display_raw(raw: np.ndarray, crop: tuple[slice, slice]) -> np.ndarray:
    image = raw[crop]
    lo, hi = np.percentile(image, [1, 99])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def _overlay(mask: np.ndarray, gt: np.ndarray, crop: tuple[slice, slice]) -> np.ndarray:
    pred = mask[crop].astype(bool)
    truth = gt[crop].astype(bool)
    rgb = np.zeros((*pred.shape, 3), dtype=float)
    rgb[..., 0] = pred
    rgb[..., 1] = truth
    rgb[..., 2] = np.logical_and(pred, truth)
    return rgb


def _plot_pruning_panel(
    full: pd.DataFrame, retained_sources: set[str], output_path: Path
) -> None:
    stacks = []
    for _, row in full.iterrows():
        stack = _read_stack(_resolve_path(row["stacked_path"]))
        stacks.append((row, (stack[1] > 0).astype(np.uint8)))
    crop = _crop_bounds([mask for _, mask in stacks], pad=30)
    tile_w, tile_h = 260, 300
    cols, rows = 4, 3
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (row, mask) in enumerate(stacks):
        col = idx % cols
        row_index = idx // cols
        x0 = col * tile_w
        y0 = row_index * tile_h
        kept = str(row["candidate_source"]) in retained_sources
        status = "kept" if kept else "removed"
        color = (25, 115, 25) if kept else (130, 30, 30)
        title = (
            f"{row['candidate_source']} | IoU {row['true_iou']:.3f} | "
            f"QA {row['qa_score']:.3f} | {status}"
        )
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + tile_h - 1], outline=(210, 210, 210))
        draw.text((x0 + 8, y0 + 8), title, fill=color)
        thumb = _mask_image(mask[crop]).resize((236, 236), Image.Resampling.NEAREST)
        canvas.paste(thumb, (x0 + 12, y0 + 48))
    canvas.save(output_path)


def _plot_vote_panel(
    retained: pd.DataFrame,
    masks: list[np.ndarray],
    raw_image: np.ndarray,
    gt_mask: np.ndarray,
    fused_mask: np.ndarray,
    normalized_vote: np.ndarray,
    output_path: Path,
) -> None:
    crop = _crop_bounds([*masks, gt_mask, fused_mask], pad=30)
    panels: list[tuple[str, Image.Image]] = [
        ("Raw crop", _gray_image(_display_raw(raw_image, crop))),
        ("SEG ground-truth mask", _mask_image(gt_mask[crop])),
    ]
    for offset, ((_, row), mask) in enumerate(
        zip(retained.iterrows(), masks, strict=True), start=2
    ):
        panels.append(
            (
                f"{row['candidate_source']}\nIoU {row['true_iou']:.3f}, w {row[WEIGHT_COLUMN]:.3f}",
                _rgb_image(_overlay(mask, gt_mask, crop)),
            )
        )
    panels.append(("Weighted vote / sum(w)", _vote_image(normalized_vote[crop])))
    panels.append(
        (
            "Final fused output\nred=pred, green=SEG, cyan=overlap",
            _rgb_image(_overlay(fused_mask, gt_mask, crop)),
        )
    )
    _panel_grid(panels, output_path, cols=4, tile_size=(310, 340), image_size=(286, 286))


def _plot_weight_bar(retained: pd.DataFrame, output_path: Path) -> None:
    width, height = 940, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 15), "Candidate weights used by the ENSEMBLE vote", fill=(0, 0, 0))
    left, top, bottom = 70, 70, 300
    draw.line([left, top, left, bottom], fill=(0, 0, 0))
    draw.line([left, bottom, width - 30, bottom], fill=(0, 0, 0))
    names = retained["candidate_source"].astype(str).tolist()
    raw = retained[RAW_WEIGHT_COLUMN].astype(float).tolist()
    weights = retained[WEIGHT_COLUMN].astype(float).tolist()
    group_w = 210
    bar_w = 55
    for idx, name in enumerate(names):
        x = left + 35 + idx * group_w
        raw_h = int((bottom - top) * min(raw[idx], 1.0))
        weight_h = int((bottom - top) * min(weights[idx], 1.0))
        draw.rectangle([x, bottom - raw_h, x + bar_w, bottom], fill=(70, 120, 180))
        draw.rectangle(
            [x + bar_w + 15, bottom - weight_h, x + 2 * bar_w + 15, bottom],
            fill=(220, 120, 50),
        )
        draw.text((x - 10, bottom + 10), name, fill=(0, 0, 0))
        draw.text((x, bottom - raw_h - 16), f"{raw[idx]:.3f}", fill=(0, 0, 0))
        draw.text(
            (x + bar_w + 15, bottom - weight_h - 16),
            f"{weights[idx]:.3f}",
            fill=(0, 0, 0),
        )
    draw.rectangle([720, 40, 740, 55], fill=(70, 120, 180))
    draw.text((750, 38), "Deep Sets raw", fill=(0, 0, 0))
    draw.rectangle([720, 65, 740, 80], fill=(220, 120, 50))
    draw.text((750, 63), "cluster softmax weight", fill=(0, 0, 0))
    image.save(output_path)


def _gray_image(values: np.ndarray) -> Image.Image:
    arr = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _mask_image(mask: np.ndarray) -> Image.Image:
    arr = mask.astype(bool).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L").convert("RGB")


def _rgb_image(values: np.ndarray) -> Image.Image:
    arr = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _vote_image(values: np.ndarray) -> Image.Image:
    v = np.clip(values, 0, 1)
    rgb = np.zeros((*v.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * v).astype(np.uint8)
    rgb[..., 1] = (180 * np.sqrt(v)).astype(np.uint8)
    rgb[..., 2] = (255 * (1 - v)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _panel_grid(
    panels: list[tuple[str, Image.Image]],
    output_path: Path,
    *,
    cols: int,
    tile_size: tuple[int, int],
    image_size: tuple[int, int],
) -> None:
    rows = math.ceil(len(panels) / cols)
    tile_w, tile_h = tile_size
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (title, image) in enumerate(panels):
        col = idx % cols
        row = idx // cols
        x0, y0 = col * tile_w, row * tile_h
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + tile_h - 1], outline=(210, 210, 210))
        draw.text((x0 + 8, y0 + 8), title, fill=(0, 0, 0))
        resized = image.resize(image_size, Image.Resampling.NEAREST)
        canvas.paste(resized, (x0 + 12, y0 + 48))
    canvas.save(output_path)


def _fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    return str(value)


def _tex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _latex_table(
    frame: pd.DataFrame, columns: list[str], headers: list[str], digits: int = 4
) -> str:
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule"]
    lines.append(" & ".join(_tex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for _, row in frame.iterrows():
        lines.append(
            " & ".join(_tex_escape(_fmt(row[c], digits)) for c in columns) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _metadata_value(metadata: pd.DataFrame, name: str) -> str:
    return str(metadata.loc[metadata["name"].eq(name), "value"].iloc[0])


def _build_tex(
    *,
    retained: pd.DataFrame,
    full: pd.DataFrame,
    retained_table: pd.DataFrame,
    full_table: pd.DataFrame,
    pixel_table: pd.DataFrame,
    metadata: pd.DataFrame,
    eval_row: pd.Series,
    payload: dict,
) -> str:
    best_candidate = float(full["true_iou"].max())
    best_retained = float(retained["true_iou"].max())
    fused_iou = float(eval_row["jaccard"])
    fused_f1 = float(eval_row["f1"])
    weight_sum = float(retained[WEIGHT_COLUMN].sum())
    config = payload["config"]
    categories = payload["preprocess"]["categories"]
    one_hot = sum(len(values) for values in categories.values())

    full_short = full_table[
        [
            "rank_by_pairwise",
            "candidate_source",
            "status",
            "true_iou",
            "qa_score",
            "pairwise_cluster_score",
            "pairwise_cluster_rank",
        ]
    ].head(12)
    retained_short = retained_table[
        [
            "candidate_source",
            "true_iou",
            "qa_score",
            "pairwise_cluster_score",
            RAW_WEIGHT_COLUMN,
            WEIGHT_COLUMN,
            "effective_boundary_support",
            "normalized_vote_weight",
        ]
    ]
    pixel_short = pixel_table[
        [
            "object",
            "mask_area_px",
            "intersection_px",
            "union_px",
            "iou",
            "f1",
            "boundary_px",
            "interior_px",
        ]
    ].iloc[:4]

    full_table_tex = _latex_table(
        full_short,
        [
            "rank_by_pairwise",
            "candidate_source",
            "status",
            "true_iou",
            "qa_score",
            "pairwise_cluster_score",
            "pairwise_cluster_rank",
        ],
        ["rank", "source", "status", "true IoU", "QA", "pairwise", "cluster rank"],
        digits=3,
    )
    retained_table_tex = _latex_table(
        retained_short,
        [
            "candidate_source",
            "true_iou",
            "qa_score",
            "pairwise_cluster_score",
            RAW_WEIGHT_COLUMN,
            WEIGHT_COLUMN,
            "effective_boundary_support",
            "normalized_vote_weight",
        ],
        ["source", "true IoU", "QA", "pairwise", "raw DS", "vote w", "bnd support", "w/sum"],
        digits=6,
    )
    pixel_table_tex = _latex_table(
        pixel_short,
        [
            "object",
            "mask_area_px",
            "intersection_px",
            "union_px",
            "iou",
            "f1",
            "boundary_px",
            "interior_px",
        ],
        ["object", "area", "intersect", "union", "IoU", "F1", "boundary", "interior"],
        digits=3,
    )

    feature_examples = retained_table[
        [
            "candidate_source",
            "training_target_for_this_cluster",
            "softmax_formula_value",
            "local_image_support_score",
            "effective_boundary_support",
        ]
    ]
    geometry_support = retained[
        [
            "candidate_source",
            "mask_area_pixels",
            "mask_extent",
            "full_mask_area_log_ratio_to_cluster_median",
            "rect_mask_area_log_ratio_to_cluster_median",
            "ranker_cluster_full_boundary_contrast_rank",
            "ranker_cluster_rect_boundary_contrast_rank",
            "ranker_cluster_full_raw_contrast_norm_rank",
            "ranker_cluster_rect_raw_contrast_norm_rank",
            "local_image_support_score",
        ]
    ].copy()
    feature_examples_tex = _latex_table(
        feature_examples,
        [
            "candidate_source",
            "training_target_for_this_cluster",
            "softmax_formula_value",
            "local_image_support_score",
            "effective_boundary_support",
        ],
        ["source", "training target", "exp(raw-max)", "local support", "used support"],
        digits=6,
    )
    geometry_support_tex = _latex_table(
        geometry_support,
        [
            "candidate_source",
            "mask_area_pixels",
            "mask_extent",
            "full_mask_area_log_ratio_to_cluster_median",
            "rect_mask_area_log_ratio_to_cluster_median",
            "ranker_cluster_full_boundary_contrast_rank",
            "ranker_cluster_rect_boundary_contrast_rank",
            "ranker_cluster_full_raw_contrast_norm_rank",
            "ranker_cluster_rect_raw_contrast_norm_rank",
            "local_image_support_score",
        ],
        [
            "source",
            "area",
            "extent",
            "full log area ratio",
            "rect log area ratio",
            "full boundary rank",
            "rect boundary rank",
            "full contrast rank",
            "rect contrast rank",
            "local support",
        ],
        digits=4,
    )

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{float}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{longtable}}

\graphicspath{{{{qa_pruned_ensemble_real_case_2026_06_03_assets/}}}}

\title{{A Real Inference Walkthrough of the QA-pruned ENSEMBLE}}
\author{{silver-truth notes}}
\date{{June 3, 2026}}

\begin{{document}}
\maketitle

\section{{What this document is}}

This is not a toy example.  It follows one real test object from the current best strict
QA-pruned ENSEMBLE run.  The object is:

\begin{{itemize}}
\item Dataset: \texttt{{Fluo-C2DL-MSC}}
\item Fold: \texttt{{fold-1}}
\item Split: test
\item Test campaign: \texttt{{02}}
\item Frame: \texttt{{t011}}
\item SEG object label evaluated here: \texttt{{1}}
\end{{itemize}}

This run is the stability-selected strict QA/ENSEMBLE row
\texttt{{validation\_mean\_minus\_2p000std\_selected}}.  Its full strict-panel mean
IoU is {_fmt(float(_metadata_value(metadata, "selected_strict_mean_iou")), 6)}
and mean F1 is {_fmt(float(_metadata_value(metadata, "selected_strict_mean_f1")), 6)}.
For this one cell, the final fused output reaches IoU {_fmt(fused_iou, 6)}
and F1 {_fmt(fused_f1, 6)}.  The best upstream candidate has IoU
{_fmt(best_candidate, 6)}; the best retained candidate also has IoU
{_fmt(best_retained, 6)}.  So this is a nice example where the ENSEMBLE is
not merely selecting the best mask.  It produces a mask better than any single
input candidate.

\section{{Pipeline in one picture}}

\[
\text{{competitor segmentations}}
\rightarrow
\text{{synchronization by TRA/SEG metadata}}
\rightarrow
\text{{QA/ranker/support features}}
\rightarrow
\text{{QA-pruned candidate set}}
\rightarrow
\text{{Deep Sets ENSEMBLE weights}}
\rightarrow
\text{{boundary-aware weighted vote}}
\rightarrow
\text{{final mask}}.
\]

The important distinction for the paper terminology: this document is about our
\textbf{{ENSEMBLE}} method.  It is not calling SIMPLE or the external fusion
baselines our method.

\section{{The synchronized object cluster}}

Synchronization gives us the object identity used for training and evaluation:
campaign \texttt{{02}}, frame \texttt{{t011}}, label \texttt{{1}}.  The important
wording correction is that there are two GT products in the Cell Tracking Challenge
layout:
\[
\texttt{{TRA/man\_track011.tif}} \quad\text{{and}}\quad
\texttt{{SEG/man\_seg011.tif}}.
\]
TRA is the tracking marker/lineage mask.  SEG is the segmentation ground truth.
The supervised QA target and the object-vote evaluation mask use the SEG object
mask, not the TRA marker mask.  For this exact object, the stored stack channel
used as ground truth is identical to
\texttt{{02\_GT/SEG/man\_seg011.tif}} for label 1 (IoU 1.000000) and is not
identical to \texttt{{02\_GT/TRA/man\_track011.tif}} for label 1 (IoU 0.343050).

Each competitor can propose a mask for this same synchronized object.  That is
why the rows below can all be compared to the same SEG ground-truth object.
In plain language: the SEG mask is the referee holding up the same target cell
while every competitor says, ``I think it looks like this.''

The upstream cluster contains {len(full)} candidate masks.  The QA-pruned candidate
pool keeps {len(retained)} of them for the ENSEMBLE vote and removes the other
{len(full) - len(retained)} before voting.

\begin{{table}}[H]
\centering
\caption{{All upstream candidates for the real object.  The status column shows
which segmentations enter the ENSEMBLE vote.}}
\resizebox{{\textwidth}}{{!}}{{%
{full_table_tex}
}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{candidate_pruning_panel.png}}
\caption{{The twelve real candidate segmentations in the upstream object cluster.
Green titles are retained; red titles are removed before the ENSEMBLE vote.}}
\end{{figure}}

\section{{Feature calculation before the ENSEMBLE}}

The feature table is built before inference.  For each candidate, it stores three
kinds of evidence.

\paragraph{{1. QA evidence.}}
The column \texttt{{qa\_score}} is the QA model's predicted Jaccard/IoU for this
candidate crop.  During QA training, the input is a two-channel crop
\[
X \in \mathbb{{R}}^{{2 \times H \times W}},
\]
where channel 0 is the image crop and channel 1 is the candidate mask.  The target is
the true Jaccard score against the SEG ground-truth mask:
\[
\operatorname{{IoU}}(M,G)=\frac{{|M\cap G|}}{{|M\cup G|}}.
\]
The QA network is a regression model, not a binary classifier.

\paragraph{{2. Support evidence.}}
The support features ask whether the image and neighboring information agree with
the proposed mask.  The local image support used by this final vote is
\texttt{{local\_image\_support\_score}}.  In this feature set it is constructed as
the average of four cluster-relative ranks:
\[
s_\text{{local}} =
\operatorname{{mean}}\left(
r_\text{{full boundary}},
r_\text{{rect boundary}},
r_\text{{full contrast}},
r_\text{{rect contrast}}
\right),
\]
with missing values filled as 0.5 and clipped to $[0,1]$.  The ranks are computed
inside the candidate cluster: a larger value means this candidate has stronger
local image support relative to the other suggestions for the same cell.

For boundary-aware voting, local support is not allowed to fall below 0.75:
\[
s_i^\ast = \operatorname{{clip}}(s_i, 0.75, 1.0).
\]
This is not a hard rejection.  It says: if a candidate has weak local support,
trust its boundary pixels less, but do not completely erase the candidate.

\paragraph{{3. Geometry and context evidence.}}
Geometry features describe the proposed mask: area, area fraction, bounding-box
aspect ratio, extent, perimeter/area ratio, component count, border-touch fraction,
centroid offset, and log area ratio to the cluster median.  The model also receives
cluster-relative versions of many features:
\[
\operatorname{{rank}}_c(x_i), \quad
x_i-\bar x_c, \quad
\max_j x_j - x_i.
\]
This is the ``don't judge a mask alone at lunch'' part: a cell mask can look
reasonable in isolation, but strange compared with the other proposals for the
same object.

For this object, some of the concrete geometry/support values are:

\begin{{table}}[H]
\centering
\caption{{Actual geometry and support features for the retained candidates.
Area is foreground pixels in the candidate crop.  Extent is
$\text{{area}}/\text{{bounding-box area}}$.  Log area ratios compare the candidate
area to the median area of this synchronized object cluster.  The four rank columns
are the inputs averaged into \texttt{{local\_image\_support\_score}}.}}
\resizebox{{\textwidth}}{{!}}{{%
{geometry_support_tex}
}}
\end{{table}}

\section{{QA network architecture}}

The QA model used to produce the QA score is implemented as a two-channel Jaccard
regressor:

\[
X \in \mathbb{{R}}^{{B\times 2\times H\times W}}
\rightarrow
\text{{ImageNet backbone}}
\rightarrow
\text{{dropout}}
\rightarrow
\text{{linear scalar head}}
\rightarrow
\sigma(\cdot)
\rightarrow
\hat q \in [0,1]^{{B\times 1}}.
\]

For the ResNet-18 configuration, the first convolution is changed from RGB to two
input channels:
\[
\operatorname{{Conv2d}}(2,64,k=7,s=2,p=3),
\]
the ImageNet classification head is replaced by identity, dropout uses
$p=0.3$, and the final head is
\[
\operatorname{{Linear}}(512,1)
\]
followed by a sigmoid.  The loss is mean squared error against true candidate
IoU/Jaccard.  The practical output is the scalar \texttt{{qa\_score}} shown in
the tables.

\section{{Deep Sets ENSEMBLE architecture}}

The final ENSEMBLE does not use an ordinary per-row MLP.  It uses a Deep Sets
regressor so that each candidate can be scored in the context of the other
candidates for the same object.

For this fold, the stored payload says:
\begin{{itemize}}
\item numeric input features: {len(payload["feature_columns"])}
\item categorical columns: {len(payload["categorical_columns"])}
      (\texttt{{dataset}}, \texttt{{candidate\_source}})
\item one-hot categorical features: {one_hot}
\item total network input dimension: {payload["input_dim"]}
\item hidden width: {config["deep_sets_hidden_dim"]}
\item training rows: {payload["train_rows"]}
\item training clusters: {payload["train_clusters"]}
\item training epochs: {config["deep_sets_epochs"]}
\item learning rate: {config["deep_sets_learning_rate"]}
\item weight decay: {config["deep_sets_weight_decay"]}
\item final training loss: {_fmt(float(payload["train_losses"][-1]), 8)}
\end{{itemize}}

For a cluster with candidates $i=1,\ldots,n$, each candidate has a feature vector
$x_i\in\mathbb{{R}}^{{250}}$.  The network is:
\[
\phi(x_i) =
\operatorname{{ReLU}}\left(W_2\,
\operatorname{{ReLU}}(W_1 x_i + b_1)+b_2\right),
\quad \phi(x_i)\in\mathbb{{R}}^{{64}}.
\]
\[
c = \frac{{1}}{{n}}\sum_i \phi(x_i),
\]
\[
r_i =
\sigma\left(
W_4\,\operatorname{{ReLU}}\left(W_3[\phi(x_i),c]+b_3\right)+b_4
\right).
\]

The shapes are:
\[
X_c: n\times 250,\quad
\phi(X_c): n\times 64,\quad
c:1\times 64,\quad
[\phi(X_c),c]:n\times 128,\quad
r:n\times 1.
\]

The training target for each candidate is a softened oracle quality:
\[
y_i=\exp\left(\frac{{\operatorname{{IoU}}_i-\max_j\operatorname{{IoU}}_j}}{{0.08}}\right).
\]
The best candidate in a training cluster gets target 1.  Candidates far below
the best get values close to 0.  This is a regression target, so the network is
being trained to assign relative usefulness, not to make a yes/no decision.

\begin{{table}}[H]
\centering
\caption{{Retained candidates and their final ENSEMBLE weights.}}
\resizebox{{\textwidth}}{{!}}{{%
{retained_table_tex}
}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{The two equations that turn true-IoU training supervision and raw Deep
Sets output into numbers used in this example.}}
\resizebox{{0.92\textwidth}}{{!}}{{%
{feature_examples_tex}
}}
\end{{table}}

\section{{Weight calibration and pixel vote}}

The Deep Sets raw output $r_i$ is converted into the stored vote weight by a
cluster softmax-style transform with temperature 1:
\[
w_i=\exp\left(\operatorname{{clip}}(r_i-\max_j r_j,-50,0)\right).
\]

For this object:
\[
\sum_i w_i = {_fmt(weight_sum, 6)}.
\]

The boundary-aware vote is computed pixel by pixel.  Let $M_i(p)\in\{{0,1\}}$ be
candidate $i$'s mask at pixel $p$.  Let $B_i(p)$ mark the one-pixel boundary of
candidate $i$ after one 3-by-3 erosion.  The pixel contribution is:
\[
V(p)=
\sum_i
M_i(p)\,
w_i\,
\begin{{cases}}
s_i^\ast, & p\in B_i,\\
1, & p\notin B_i.
\end{{cases}}
\]

The final ENSEMBLE mask is:
\[
\hat M(p)=
\mathbb{{1}}\left[
\frac{{V(p)}}{{\sum_i w_i}}\ge 0.48
\right].
\]

This is why a candidate can be kept but still softened.  Interior pixels carry
the full candidate weight.  Boundary pixels carry a support-discounted weight.
It is like asking each competitor to vote, but making them speak a little quieter
around uncertain edges.

\begin{{table}}[H]
\centering
\caption{{Pixel accounting for the retained masks and final fused output.}}
\resizebox{{\textwidth}}{{!}}{{%
{pixel_table_tex}
}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{retained_vote_panel.png}}
\caption{{The retained masks, weighted vote map, and final fused output.
In RGB overlays, red means predicted foreground, green means SEG ground truth,
and cyan/white-ish overlap means agreement.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.74\textwidth]{{candidate_weight_bar.png}}
\caption{{Raw Deep Sets outputs and calibrated vote weights.  The bad
\texttt{{MU-Akb-CZ}} candidate enters the retained pool because its QA score is high,
but the Deep Sets model assigns it an almost-zero raw score before calibration.}}
\end{{figure}}

\section{{What is happening in this example}}

The awkward but useful lesson is that the QA score alone is not enough.  The
\texttt{{MU-Akb-CZ}} candidate has QA {_fmt(float(retained.loc[retained["candidate_source"].eq("MU-Akb-CZ"), "qa_score"].iloc[0]), 6)}
but true IoU only {_fmt(float(retained.loc[retained["candidate_source"].eq("MU-Akb-CZ"), "true_iou"].iloc[0]), 6)}.
So the QA-pruned candidate-selection stage lets a false-positive candidate into
the retained set.  The newer
Deep Sets ENSEMBLE catches much of that mistake by using cluster context,
geometry, source identity, image support, and relative ranks.  Its raw score for
that candidate is only {_fmt(float(retained.loc[retained["candidate_source"].eq("MU-Akb-CZ"), RAW_WEIGHT_COLUMN].iloc[0]), 8)}.

The final mask is better than any input because the two good retained candidates
do not make exactly the same boundary errors.  The weighted vote keeps pixels
where the good masks agree and suppresses some boundary noise.  The ENSEMBLE is
therefore doing more than picking the best competitor; it is combining compatible
parts of the retained masks.

\section{{How bad is the QA failure?}}

This example is not cherry-picked to make QA look good.  It actually exposes one
of the main weaknesses of the original QA idea.  The \texttt{{MU-Akb-CZ}} candidate
has the highest QA score in the upstream cluster (0.841006) but a true IoU of only
0.343124.  That is a spectacular local failure.

Across the ten selected test folds, using the full upstream candidate pool, the QA
score has Pearson correlation 0.4626 and Spearman correlation 0.5777 with true IoU.
Pearson measures linear agreement between predicted and true quality.  Spearman
measures rank agreement: if QA says candidate A is better than B, does true IoU
usually agree?  Spearman is the more relevant number for pruning and candidate
selection because we mostly care about ordering candidates inside each object
cluster.

The problem is dataset-dependent.  On \texttt{{Fluo-C2DL-MSC}}, QA is actively
misleading on the test split: Pearson is -0.2725 and Spearman is -0.2922 in the
full upstream pool.  In the retained QA-pruned pool, the same dataset is still
negative: Pearson -0.2177 and Spearman -0.2189.  So the low QA scores on the good
MSC candidates in this example are not a visualization oddity.  They are a real
domain-shift symptom.

\begin{{table}}[H]
\centering
\caption{{QA and downstream-score behavior on the selected test folds.}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrr}}
\toprule
score used to pick top candidate & exact oracle rate & gap $\geq$ 0.20 & bad top, IoU $<$ 0.5, oracle $\geq$ 0.8 & mean gap \\
\midrule
QA score, all datasets & 0.2015 & 0.1500 & 0.0328 & 0.0969 \\
QA score, Fluo-C2DL-MSC only & 0.0965 & 0.4430 & 0.1667 & 0.2124 \\
Pairwise/context score, all datasets & 0.4907 & 0.0458 & 0.0102 & 0.0351 \\
Pairwise/context score, Fluo-C2DL-MSC only & 0.3333 & 0.1798 & 0.0658 & 0.1070 \\
Deep Sets raw score on retained pool, all datasets & 0.3945 & 0.0317 & 0.0028 & 0.0384 \\
Deep Sets raw score on retained pool, Fluo-C2DL-MSC only & 0.4868 & 0.0439 & 0.0044 & 0.0434 \\
\bottomrule
\end{{tabular}}
}}
\end{{table}}

The interpretation is blunt: QA alone is not reliable enough to be the main
decision mechanism, especially for MSC.  The context/pairwise/ranker features and
the Deep Sets ENSEMBLE are doing much of the rescue.  QA is still part of the
feature set and is useful on several datasets, but this case argues against
claiming that QA alone is the hero.  The more defensible claim is that the method
uses QA as one imperfect signal inside a broader ENSEMBLE and candidate-ranking
system.

\section{{Where this can still fail}}

The same machinery can fail when the retained set is dominated by mutually
inconsistent masks, when the best mask is pruned before the vote, or when the
Deep Sets weights are too smooth after the cluster-softmax transform.  Notice
that even an almost-zero raw Deep Sets score becomes
$\exp(-1)\approx 0.3679$ if the best raw score is near 1 and the temperature is
1.  That is deliberate smoothing, but it also means a bad retained candidate is
not fully silenced.  This is one reason the result is strong but still not a
magic wand.

\section{{Files used}}

\begin{{itemize}}
\item Candidate parquet:
\texttt{{{_tex_escape(CANDIDATE_PARQUET.relative_to(ROOT))}}}
\item Upstream full candidate parquet:
\texttt{{{_tex_escape(FULL_CANDIDATE_PARQUET.relative_to(ROOT))}}}
\item Object-vote directory:
\texttt{{{_tex_escape(EVAL_DIR.relative_to(ROOT))}}}
\item Deep Sets model payload:
\texttt{{{_tex_escape(MODEL_PAYLOAD.relative_to(ROOT))}}}
\item Generated CSV tables live in:
\texttt{{{_tex_escape(OUT_DIR.relative_to(ROOT))}}}
\end{{itemize}}

\end{{document}}
"""


if __name__ == "__main__":
    main()
