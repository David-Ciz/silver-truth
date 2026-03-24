#!/usr/bin/env python3
"""
Ablation pipeline runner.

Reads a variant YAML config (with optional ``extends: base`` inheritance), resolves
all path templates, and runs the full experiment sequence for one fold.  Completed
steps are checkpointed in ``.state/<run_id>.json`` so re-running resumes safely.

Quick reference
---------------
# Run baseline, fold 1
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1

# Dry-run (print commands without executing)
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1 --dry-run

# Resume after training finished (skips already-completed steps automatically)
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1

# Different QA model
python scripts/run_ablation.py --config experiments/variants/resnet18_qa.yaml --fold 1

# Different dataset
python scripts/run_ablation.py --config experiments/variants/fluo_dataset.yaml --fold 1

# Run both folds in parallel (two terminals)
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1 &
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 2
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import mlflow
import yaml
from mlflow.tracking import MlflowClient

from silver_truth.experiment_tracking import (
    ABLATION_CONFIG_ENV,
    ABLATION_EXPERIMENT_ENV,
    ABLATION_PARENT_RUN_ENV,
    ABLATION_ROOT_RUN_ENV,
    ABLATION_RUN_KEY_ENV,
    ABLATION_TRACKING_URI_ENV,
    resolve_mlflow_tracking_uri,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
STATE_DIR = PROJECT_ROOT / ".state"
ABLATION_DATA_ROOT_ENV = "ABLATION_DATA_ROOT"
ABLATION_OUTPUT_DIR_ENV = "ABLATION_OUTPUT_DIR"

# Base directory that mirrors training._checkpoint_path — used when no explicit
# --checkpoints-dir is passed to ensemble-experiment.
_DEFAULT_CKPT_BASE = PROJECT_ROOT / "data/ensemble_data/results/checkpoints"
_DEFAULT_DATABANKS_DIR = PROJECT_ROOT / "data/ensemble_data/databanks"


def _normalise_split_name(split: str) -> str:
    split_str = str(split).strip().lower()
    if split_str in {"1", "fold-1"}:
        return "fold-1"
    if split_str in {"2", "fold-2"}:
        return "fold-2"
    if split_str == "mixed":
        return "mixed"
    raise ValueError(f"Unsupported split value: {split}")


def _databank_dir(
    paper_runs: str, variant: str, split_name: str, tag: str = "unfiltered"
) -> str:
    """
    Return a per-experiment databank output directory under paper_runs so that
    each variant/fold/tag combination gets its own isolated directory and files
    never overwrite each other.

    Example: data/paper_runs/ensemble/baseline/fold-1/databank_unfiltered
    """
    return f"{paper_runs}/ensemble/{variant}/{split_name}/databank_{tag}"


def _databank_parquet(databank_dir: str, dataset: str, version: str = "C1") -> str:
    """
    Reproduce the parquet filename written by build-databank inside the given dir.
    Pattern: {version}_{ds_code}-42-7015_QA--.parquet
    """
    _DS_CODES = {"BF-C2DL-HSC": "ds1", "BF-C2DL-MuSC": "ds2"}
    ds_code = _DS_CODES.get(dataset, dataset)
    name = f"{version}_{ds_code}-42-7015_QA--"
    return f"{databank_dir}/{name}.parquet"


def _ckpt_dir(
    paper_runs: str, variant: str, split_name: str, tag: str = "baseline"
) -> str:
    """Return a per-experiment checkpoint directory."""
    return f"{paper_runs}/ensemble/{variant}/{split_name}/checkpoints_{tag}"


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _resolve_extends_path(current_path: Path, extends: str) -> Path:
    if extends == "base":
        return EXPERIMENTS_DIR / "base.yaml"

    candidate = Path(extends)
    if candidate.is_absolute():
        return candidate

    for resolved in (
        (current_path.parent / candidate).resolve(),
        (EXPERIMENTS_DIR / candidate).resolve(),
    ):
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Could not resolve extends target '{extends}' from {current_path}"
    )


def _load_config_with_inheritance(path: Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    extends = raw.pop("extends", None)
    if not extends:
        return raw

    parent_path = _resolve_extends_path(path, str(extends))
    parent = _load_config_with_inheritance(parent_path)
    return {**parent, **raw}


def load_config(variant_path: Path, fold: str) -> dict[str, Any]:
    """
    Load a variant YAML, merging with ``base.yaml`` when ``extends: base`` is set.
    Inject runtime values (project_root, fold, variant name) and resolve all
    ``{…}`` template strings.
    """
    config = _load_config_with_inheritance(variant_path)

    # Inject runtime values.
    config["project_root"] = str(PROJECT_ROOT)
    runtime_root = Path(
        os.getenv(ABLATION_DATA_ROOT_ENV, str(PROJECT_ROOT))
    ).expanduser()
    paper_runs_root = Path(
        os.getenv(ABLATION_OUTPUT_DIR_ENV, str(PROJECT_ROOT / "data/paper_runs"))
    ).expanduser()
    config["runtime_root"] = str(runtime_root.resolve())
    config["paper_runs_root"] = str(paper_runs_root.resolve())
    if os.getenv("MLFLOW_TRACKING_URI"):
        config["mlflow_tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    split_name = _normalise_split_name(fold)
    config["fold"] = split_name
    config["split_name"] = split_name
    config["variant"] = variant_path.stem  # e.g. "baseline", "resnet18_qa"

    # Resolve template strings (two passes to handle nested references).
    for _ in range(2):
        config = _resolve_templates(config, config)

    return config


def _resolve_templates(obj: Any, ctx: dict[str, Any]) -> Any:
    """Recursively resolve ``{key}`` placeholders using *ctx* as the namespace."""
    if isinstance(obj, str):
        try:
            return obj.format_map(ctx)
        except (KeyError, ValueError):
            return obj
    if isinstance(obj, dict):
        return {k: _resolve_templates(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_templates(v, ctx) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Step definitions
# ─────────────────────────────────────────────────────────────────────────────

# A step is a dict with keys:
#   id          – unique snake_case identifier (used in checkpoint key)
#   phase       – "A", "B", or "C"
#   name        – human-readable label
#   cmd         – shell command string (already resolved)
#   wait        – (optional) if True, print WAIT banner and stop; resume resumes here


def build_steps(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the ordered list of steps from a resolved config."""
    fold = cfg["fold"]
    split_name = cfg["split_name"]
    variant = cfg["variant"]
    dataset = cfg["dataset"]
    crop_size = cfg["crop_size"]
    threshold = cfg["qa_threshold"]
    threshold_sweep = [float(t) for t in cfg.get("qa_threshold_sweep", [threshold])]
    mlflow_uri = cfg["mlflow_tracking_uri"]
    fusion_models = " ".join(f"--models {m}" for m in cfg["fusion_models"])
    ensemble_version = cfg.get("ensemble_version", "C1")
    ensemble_augmentation = cfg.get("ensemble_augmentation", "basic")
    ensemble_encoder_name = cfg.get("ensemble_encoder_name", "resnet34")
    ensemble_encoder_weights = cfg.get("ensemble_encoder_weights")
    ensemble_init_checkpoint = cfg.get("ensemble_init_checkpoint")
    ensemble_encoder_weights_arg = (
        f"  --encoder-weights {ensemble_encoder_weights}"
        if ensemble_encoder_weights
        else ""
    )
    ensemble_init_checkpoint_arg = (
        f"  --init-from-checkpoint {ensemble_init_checkpoint}"
        if ensemble_init_checkpoint
        else ""
    )

    def _threshold_label(value: float) -> str:
        return f"{value:.2f}"

    # Shorthand path helpers
    qa_parquet = cfg["qa_parquet_template"]
    whole_image_parquet = cfg["whole_image_parquet_template"]
    paper_ready = cfg["paper_ready_parquet_template"]
    paper_base = paper_ready.replace("_paper_ready.parquet", "_paper_base.parquet")
    qa_model = cfg["qa_model_template"]
    qa_excel = cfg["qa_excel_template"]

    fusion_out = cfg["fusion_output_template"]
    ablation_out = cfg["ablation_output_template"]

    paper_runs = cfg["paper_runs_root"]
    competitor_csv = f"{paper_runs}/baselines/{dataset}_{split_name}_competitors.csv"
    compare_experiment = f"paper-compare-{dataset}-{variant}-{split_name}"
    phasea_fusion_experiment = (
        f"phaseA-fusion-baselines-{dataset}-{variant}-{split_name}"
    )
    phaseb_qa_train_experiment = f"phaseB-qa-train-{dataset}-{variant}-{split_name}"
    phaseb_qa_regression_experiment = (
        f"phaseB-qa-regression-{dataset}-{variant}-{split_name}"
    )
    phaseb_qa_filtering_experiment = (
        f"phaseB-qa-thresholding-{dataset}-{variant}-{split_name}"
    )

    steps: list[dict[str, Any]] = []

    # ── Phase A ──────────────────────────────────────────────────────────────

    if cfg.get("run_phaseA_competitors", True):
        steps.append(
            {
                "id": "phaseA_competitor_baseline",
                "phase": "A",
                "name": "Competitor baselines (full-image IoU/F1)",
                "cmd": (
                    f"mkdir -p {paper_runs}/baselines && "
                    f"silver-evaluation evaluate-competitor "
                    f"  {whole_image_parquet} "
                    f"  --output {competitor_csv} "
                    f"  --mlflow-experiment {compare_experiment} "
                    f"  --mlflow-run-name competitor"
                ),
            }
        )

    if cfg.get("run_phaseA_silver_truth", True):
        steps.append(
            {
                "id": "phaseA_silver_truth_reference",
                "phase": "A",
                "name": "Silver-truth reference (full-image IoU/F1)",
                "cmd": (
                    f"silver-evaluation evaluate-competitor "
                    f"  {whole_image_parquet} "
                    f"  --competitor SILVER-TRUTH "
                    f"  --output {paper_runs}/baselines/{dataset}_{split_name}_silver_truth.csv "
                    f"  --mlflow-experiment {compare_experiment}"
                ),
            }
        )

    if cfg.get("run_phaseA_fusion", True):
        steps.append(
            {
                "id": "phaseA_fusion_baseline",
                "phase": "A",
                "name": "Java fusion baselines — whole-image (IoU/F1 per model)",
                "cmd": (
                    f"mkdir -p {fusion_out} && "
                    f"python {PROJECT_ROOT}/scripts/run_fusion_experiment.py "
                    f"  --dataset {dataset} "
                    f"  --parquet-file {whole_image_parquet} "
                    f"  {'  '.join(f'--models {m}' for m in cfg['fusion_models'])} "
                    f"  --output-dir {fusion_out} "
                    f"  --mlflow-experiment {phasea_fusion_experiment} "
                    f"  --mlflow-run-name fusion_baseline "
                    f"  --mlflow-tracking-path {mlflow_uri.replace('file:', '')}"
                ),
            }
        )

    _baseline_db_dir = _databank_dir(paper_runs, variant, split_name, tag="unfiltered")
    _baseline_db_parquet = _databank_parquet(
        _baseline_db_dir, dataset, cfg.get("ensemble_version", "C1")
    )
    _baseline_ckpt_dir = _ckpt_dir(paper_runs, variant, split_name, tag="baseline")
    _baseline_exp = f"phaseA-ensemble-{dataset}-{variant}-{split_name}"

    if cfg.get("run_phaseA_ensemble", True):
        steps.append(
            {
                "id": "phaseA_ensemble_build_databank",
                "phase": "A",
                "name": f"Build ensemble databank (unfiltered, {ensemble_version})",
                "cmd": (
                    f"silver-ensemble build-databank "
                    f"  --dataset-name {dataset} "
                    f"  --qa-parquet-path {qa_parquet} "
                    f"  --version {ensemble_version} "
                    f"  --output-dir {_baseline_db_dir}"
                ),
                "output_hint": _baseline_db_parquet,
            }
        )

        steps.append(
            {
                "id": "phaseA_ensemble_train",
                "phase": "A",
                "name": "Train ensemble model (unfiltered) — TRAINING STEP, script pauses after this",
                "cmd": (
                    f"silver-ensemble ensemble-experiment "
                    f"  --name {_baseline_exp} "
                    f"  --parquet-file {_baseline_db_parquet} "
                    f"  --model-type {cfg['ensemble_model_type']} "
                    f"  --max-epochs {cfg['ensemble_max_epochs']} "
                    f"  --dataset-version {ensemble_version} "
                    f"  --augmentation {ensemble_augmentation} "
                    f"  --encoder-name {ensemble_encoder_name}"
                    f"{ensemble_encoder_weights_arg} "
                    f"{ensemble_init_checkpoint_arg} "
                    f"  --checkpoints-dir {_baseline_ckpt_dir}"
                ),
                "wait": True,
            }
        )

        steps.append(
            {
                "id": "phaseA_ensemble_evaluate",
                "phase": "A",
                "name": "Evaluate ensemble baseline (best checkpoint)",
                "cmd": (
                    f"silver-ensemble evaluate-best-checkpoint "
                    f"  --checkpoints-dir {_baseline_ckpt_dir} "
                    f"  --databank-path {_baseline_db_parquet} "
                    f"  --split-type test "
                    f"  --dataset-version {ensemble_version} "
                    f"  --output-dir {_baseline_ckpt_dir} "
                    f"  --mlflow-experiment {compare_experiment} "
                    f"  --mlflow-run-name ensemble_baseline "
                    f"  --setup-name ensemble_baseline"
                ),
            }
        )

    # ── Phase B ──────────────────────────────────────────────────────────────

    if cfg.get("run_phaseB_qa", True):
        steps.append(
            {
                "id": "phaseB_qa_compute_jaccard",
                "phase": "B",
                "name": "Compute jaccard_score in QA parquet (prerequisite for QA training)",
                "cmd": (
                    f"silver-evaluation calculate-evaluation-metrics-cli "
                    f"  --mode cropped "
                    f"  {qa_parquet}"
                ),
            }
        )

        steps.append(
            {
                "id": "phaseB_qa_train",
                "phase": "B",
                "name": f"Train QA model ({cfg['qa_model_type']}) — WILL BLOCK UNTIL TRAINING DONE",
                "cmd": (
                    f"mkdir -p {paper_runs}/qa_models {paper_runs}/qa_results && "
                    f"silver-qa cnn train "
                    f"  --parquet-file {qa_parquet} "
                    f"  --data-root {cfg['runtime_root']} "
                    f"  --output-model {qa_model} "
                    f"  --output-excel {qa_excel} "
                    f"  --model-type {cfg['qa_model_type']} "
                    f"  --input-channels {cfg['qa_input_channels']} "
                    f"  --mlflow-experiment {phaseb_qa_train_experiment} "
                    f"  --mlflow-run-name qa_train"
                ),
                "wait": True,
            }
        )

        steps.append(
            {
                "id": "phaseB_qa_evaluate_regression",
                "phase": "B",
                "name": "Evaluate QA model (regression metrics)",
                "cmd": (
                    f"silver-evaluation evaluate-qa-model "
                    f"  {qa_excel} "
                    f"  --output-dir {paper_runs}/qa_results/{split_name}_{variant}_metrics "
                    f"  --mlflow-experiment {phaseb_qa_regression_experiment} "
                    f"  --mlflow-run-name qa_regression"
                ),
            }
        )

        steps.append(
            {
                "id": "phaseB_qa_evaluate_filtering",
                "phase": "B",
                "name": "Evaluate QA model (filtering validity)",
                "cmd": (
                    f"silver-evaluation evaluate-qa-filtering "
                    f"  {qa_excel} "
                    f"  --thresholds 0.50,0.60,0.70,0.75,0.80,0.85,0.90 "
                    f"  --output-dir {paper_runs}/qa_results/{split_name}_{variant}_filtering "
                    f"  --mlflow-experiment {phaseb_qa_filtering_experiment} "
                    f"  --mlflow-run-name qa_thresholding"
                ),
            }
        )

        steps.append(
            {
                "id": "phaseB_merge_qa_predictions",
                "phase": "B",
                "name": "Merge QA predictions into paper-ready parquet",
                "cmd": (
                    f"mkdir -p {cfg['data_root']}/qa_crops/paper_inputs && "
                    f"cp {qa_parquet} {paper_base} && "
                    f"silver-evaluation merge-qa-predictions "
                    f"  {paper_base} "
                    f"  {qa_excel} "
                    f"  --output {paper_ready}"
                ),
            }
        )

    # ── Phase C ──────────────────────────────────────────────────────────────

    ablation_modes: list[str] = cfg.get("ablation_modes", [])

    if cfg.get("run_phaseC_ablation", True) and "fusion_only" in ablation_modes:
        mode_out = ablation_out.format_map({**cfg, "mode": "fusion_only"})
        steps.append(
            {
                "id": "phaseC_fusion_only_run",
                "phase": "C",
                "name": "Phase C — fusion_only: run fusion",
                "cmd": (
                    f"mkdir -p {mode_out} && "
                    f"silver-fusion run-fusion-crops "
                    f"  --qa-parquet {paper_ready} "
                    f"  {fusion_models} "
                    f"  --output-dir {mode_out} "
                    f"  --mlflow-experiment phaseC-fusion-only-crop-eval-{dataset}-{variant}-{split_name} "
                    f"  --mlflow-run-name fusion_only "
                    f"  --mlflow-tracking-path {mlflow_uri.replace('file:', '')}"
                ),
            }
        )
        for model in cfg["fusion_models"]:
            model_lower = model.lower()
            fused_pq = (
                f"{mode_out}/{model_lower}/"
                f"{split_name}_paper_ready_{model_lower}_with_fused.parquet"
            )
            steps.append(
                {
                    "id": f"phaseC_fusion_only_eval_{model_lower}",
                    "phase": "C",
                    "name": f"Phase C — fusion_only: full-image eval ({model})",
                    "cmd": (
                        f"silver-evaluation evaluate-fusion-crops "
                        f"  {fused_pq} "
                        f"  --fused-path-column {model_lower} "
                        f"  --output-dir {mode_out}/{model_lower}/fullimage "
                        f"  --output {mode_out}/{model_lower}/fullimage_eval.csv "
                        f"  --mlflow-experiment {compare_experiment} "
                        f"  --mlflow-run-name fusion_only__{model_lower} "
                        f"  --setup-name fusion_only__{model_lower} "
                        f"  --pipeline-family fusion "
                        f"  --qa-mode fusion_only"
                    ),
                }
            )

    if cfg.get("run_phaseC_ablation", True) and "qa_only" in ablation_modes:
        mode_out = ablation_out.format_map({**cfg, "mode": "qa_only"})
        filtered_pq = f"{mode_out}/filtered.parquet"
        steps.append(
            {
                "id": "phaseC_qa_only_filter",
                "phase": "C",
                "name": "Phase C — qa_only: filter parquet (top-1 per cell)",
                "cmd": (
                    f"mkdir -p {mode_out} && "
                    f"silver-evaluation filter-parquet "
                    f"  {paper_ready} "
                    f"  --mode qa_only "
                    f"  --output {filtered_pq}"
                ),
            }
        )
        steps.append(
            {
                "id": "phaseC_qa_only_eval",
                "phase": "C",
                "name": "Phase C — qa_only: full-image eval (top-1 crops)",
                "cmd": (
                    f"silver-evaluation evaluate-fusion-crops "
                    f"  {filtered_pq} "
                    f"  --fused-path-column stacked_path "
                    f"  --output-dir {mode_out}/fullimage "
                    f"  --output {mode_out}/fullimage_eval.csv "
                    f"  --mlflow-experiment {compare_experiment} "
                    f"  --mlflow-run-name qa_only__top1 "
                    f"  --setup-name qa_only__top1 "
                    f"  --pipeline-family qa_selector "
                    f"  --qa-mode qa_only"
                ),
            }
        )

    if cfg.get("run_phaseC_ablation", True) and "full_pipeline" in ablation_modes:
        for sweep_threshold in threshold_sweep:
            threshold_label = _threshold_label(sweep_threshold)
            mode_out = ablation_out.format_map(
                {**cfg, "mode": f"full_pipeline_t{threshold_label}"}
            )
            filtered_pq = f"{mode_out}/filtered.parquet"
            steps.append(
                {
                    "id": f"phaseC_full_pipeline_filter_t{threshold_label}",
                    "phase": "C",
                    "name": f"Phase C — full_pipeline: filter parquet (t={threshold_label})",
                    "cmd": (
                        f"mkdir -p {mode_out} && "
                        f"silver-evaluation filter-parquet "
                        f"  {paper_ready} "
                        f"  --mode full_pipeline "
                        f"  --threshold {sweep_threshold} "
                        f"  --output {filtered_pq}"
                    ),
                }
            )
            for model in cfg["fusion_models"]:
                model_lower = model.lower()
                fused_pq = (
                    f"{mode_out}/{model_lower}/"
                    f"filtered_{model_lower}_with_fused.parquet"
                )
                steps.append(
                    {
                        "id": f"phaseC_full_pipeline_run_t{threshold_label}_{model_lower}",
                        "phase": "C",
                        "name": f"Phase C — full_pipeline: run fusion ({model}, t={threshold_label})",
                        "cmd": (
                            f"silver-fusion run-fusion-crops "
                            f"  --qa-parquet {filtered_pq} "
                            f"  --models {model} "
                            f"  --output-dir {mode_out} "
                            f"  --mlflow-experiment phaseC-full-pipeline-crop-eval-{dataset}-{variant}-{split_name} "
                            f"  --mlflow-run-name full_pipeline_t{threshold_label} "
                            f"  --mlflow-tracking-path {mlflow_uri.replace('file:', '')}"
                        ),
                    }
                )
                steps.append(
                    {
                        "id": f"phaseC_full_pipeline_eval_t{threshold_label}_{model_lower}",
                        "phase": "C",
                        "name": f"Phase C — full_pipeline: full-image eval ({model}, t={threshold_label})",
                        "cmd": (
                            f"silver-evaluation evaluate-fusion-crops "
                            f"  {fused_pq} "
                            f"  --fused-path-column {model_lower} "
                            f"  --output-dir {mode_out}/{model_lower}/fullimage "
                            f"  --output {mode_out}/{model_lower}/fullimage_eval.csv "
                            f"  --mlflow-experiment {compare_experiment} "
                            f"  --mlflow-run-name full_pipeline_t{threshold_label}__{model_lower} "
                            f"  --setup-name full_pipeline_t{threshold_label}__{model_lower} "
                            f"  --pipeline-family fusion "
                            f"  --qa-mode full_pipeline "
                            f"  --qa-threshold {sweep_threshold}"
                        ),
                    }
                )

    if cfg.get("run_phaseC_ablation", True) and "ensemble_only" in ablation_modes:
        mode_out = ablation_out.format_map({**cfg, "mode": "ensemble_only"})
        steps.append(
            {
                "id": "phaseC_ensemble_only_eval",
                "phase": "C",
                "name": "Phase C — ensemble_only: evaluate unfiltered ensemble (reuses Phase A checkpoint)",
                "cmd": (
                    f"mkdir -p {mode_out} && "
                    f"silver-ensemble evaluate-best-checkpoint "
                    f"  --checkpoints-dir {_baseline_ckpt_dir} "
                    f"  --databank-path {_baseline_db_parquet} "
                    f"  --split-type test "
                    f"  --dataset-version {ensemble_version} "
                    f"  --output-dir {mode_out} "
                    f"  --mlflow-experiment {compare_experiment} "
                    f"  --mlflow-run-name ensemble_only "
                    f"  --setup-name ensemble_only "
                ),
            }
        )

    if cfg.get("run_phaseC_ablation", True) and "ensemble_qa" in ablation_modes:
        for sweep_threshold in threshold_sweep:
            threshold_label = _threshold_label(sweep_threshold)
            mode_out = ablation_out.format_map(
                {**cfg, "mode": f"ensemble_qa_t{threshold_label}"}
            )
            filtered_pq = (
                f"{cfg['data_root']}/qa_crops/paper_inputs/"
                f"{split_name}_full_pipeline_t{threshold_label}.parquet"
            )
            _qa_db_dir = _databank_dir(
                paper_runs, variant, split_name, tag=f"filtered_t{threshold_label}"
            )
            _qa_db_parquet = _databank_parquet(
                _qa_db_dir, dataset, cfg.get("ensemble_version", "C1")
            )
            steps.append(
                {
                    "id": f"phaseC_ensemble_qa_filter_t{threshold_label}",
                    "phase": "C",
                    "name": f"Phase C — ensemble_qa: filter parquet (t={threshold_label})",
                    "cmd": (
                        f"silver-evaluation filter-parquet "
                        f"  {paper_ready} "
                        f"  --mode full_pipeline "
                        f"  --threshold {sweep_threshold} "
                        f"  --output {filtered_pq}"
                    ),
                }
            )
            steps.append(
                {
                    "id": f"phaseC_ensemble_qa_build_databank_t{threshold_label}",
                    "phase": "C",
                    "name": f"Phase C — ensemble_qa: build filtered databank (t={threshold_label})",
                    "cmd": (
                        f"silver-ensemble build-databank "
                        f"  --dataset-name {dataset} "
                        f"  --qa-parquet-path {filtered_pq} "
                        f"  --version {ensemble_version} "
                        f"  --output-dir {_qa_db_dir}"
                    ),
                    "output_hint": _qa_db_parquet,
                }
            )
            steps.append(
                {
                    "id": f"phaseC_ensemble_qa_eval_t{threshold_label}",
                    "phase": "C",
                    "name": f"Phase C — ensemble_qa: evaluate baseline checkpoint on filtered inputs (t={threshold_label})",
                    "cmd": (
                        f"mkdir -p {mode_out} && "
                        f"silver-ensemble evaluate-best-checkpoint "
                        f"  --checkpoints-dir {_baseline_ckpt_dir} "
                        f"  --databank-path {_qa_db_parquet} "
                        f"  --split-type test "
                        f"  --dataset-version {ensemble_version} "
                        f"  --output-dir {mode_out} "
                        f"  --mlflow-experiment {compare_experiment} "
                        f"  --mlflow-run-name ensemble_qa_t{threshold_label} "
                        f"  --setup-name ensemble_qa_t{threshold_label} "
                        f"  --qa-mode full_pipeline "
                        f"  --qa-threshold {sweep_threshold}"
                    ),
                }
            )

    if (
        cfg.get("run_phaseC_ablation", True)
        and "ensemble_qa_retrained" in ablation_modes
    ):
        mode_out = ablation_out.format_map(
            {**cfg, "mode": f"ensemble_qa_retrained_t{threshold}"}
        )
        filtered_pq = (
            f"{cfg['data_root']}/qa_crops/paper_inputs/"
            f"{split_name}_full_pipeline_t{threshold:.2f}.parquet"
        )
        _retrain_db_dir = _databank_dir(
            paper_runs, variant, split_name, tag=f"filtered_t{threshold:.2f}"
        )
        _retrain_db_parquet = _databank_parquet(
            _retrain_db_dir, dataset, cfg.get("ensemble_version", "C1")
        )
        _retrain_ckpt_dir = _ckpt_dir(
            paper_runs, variant, split_name, tag=f"retrained_t{threshold:.2f}"
        )
        _retrain_exp = f"phaseC-ensemble-qa-retrained-{dataset}-{variant}-{split_name}"
        steps.append(
            {
                "id": "phaseC_ensemble_qa_retrained_train",
                "phase": "C",
                "name": "Phase C — ensemble_qa_retrained: retrain ensemble on filtered databank — TRAINING STEP",
                "cmd": (
                    f"silver-ensemble ensemble-experiment "
                    f"  --name {_retrain_exp} "
                    f"  --parquet-file {_retrain_db_parquet} "
                    f"  --model-type {cfg['ensemble_model_type']} "
                    f"  --max-epochs {cfg['ensemble_max_epochs']} "
                    f"  --dataset-version {ensemble_version} "
                    f"  --augmentation {ensemble_augmentation} "
                    f"  --encoder-name {ensemble_encoder_name}"
                    f"{ensemble_encoder_weights_arg} "
                    f"{ensemble_init_checkpoint_arg} "
                    f"  --checkpoints-dir {_retrain_ckpt_dir}"
                ),
                "wait": True,
            }
        )
        steps.append(
            {
                "id": "phaseC_ensemble_qa_retrained_eval",
                "phase": "C",
                "name": "Phase C — ensemble_qa_retrained: evaluate",
                "cmd": (
                    f"mkdir -p {mode_out} && "
                    f"silver-ensemble evaluate-best-checkpoint "
                    f"  --checkpoints-dir {_retrain_ckpt_dir} "
                    f"  --databank-path {_retrain_db_parquet} "
                    f"  --split-type test "
                    f"  --dataset-version {ensemble_version} "
                    f"  --output-dir {mode_out} "
                    f"  --mlflow-experiment {compare_experiment} "
                    f"  --mlflow-run-name ensemble_qa_retrained_t{threshold} "
                    f"  --setup-name ensemble_qa_retrained_t{threshold} "
                    f"  --qa-mode full_pipeline "
                    f"  --qa-threshold {threshold}"
                ),
            }
        )

    return steps


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────


def _state_path(run_id: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{run_id}.json"


def load_state(run_id: str) -> dict[str, Any]:
    p = _state_path(run_id)
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return {}


def save_state(run_id: str, state: dict[str, Any]) -> None:
    p = _state_path(run_id)
    with open(p, "w") as fh:
        json.dump(state, fh, indent=2)


def mark_done(run_id: str, state: dict[str, Any], step_id: str, cmd: str) -> None:
    state[step_id] = {
        "status": "done",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cmd": cmd,
    }
    save_state(run_id, state)


def mark_waiting(run_id: str, state: dict[str, Any], step_id: str, cmd: str) -> None:
    state[step_id] = {
        "status": "waiting",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cmd": cmd,
    }
    save_state(run_id, state)


def _state_meta(state: dict[str, Any]) -> dict[str, Any]:
    meta = state.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
        state["_meta"] = meta
    return meta


def _mlflow_meta(state: dict[str, Any]) -> dict[str, Any]:
    meta = _state_meta(state)
    mlflow_meta = meta.get("mlflow")
    if not isinstance(mlflow_meta, dict):
        mlflow_meta = {}
        meta["mlflow"] = mlflow_meta
    return mlflow_meta


def _ablation_experiment_name(cfg: dict[str, Any]) -> str:
    return f"ablation-{cfg['dataset']}-{cfg['variant']}-{cfg['split_name']}"


def _build_mlflow_context(
    cfg: dict[str, Any],
    config_path: Path,
    run_id: str,
    phase: str,
    state: dict[str, Any],
) -> dict[str, str]:
    mlflow_meta = _mlflow_meta(state)
    required_keys = {
        "tracking_uri",
        "experiment_name",
        "parent_run_id",
        "parent_run_name",
    }
    if required_keys.issubset(mlflow_meta):
        context = {
            "tracking_uri": str(mlflow_meta["tracking_uri"]),
            "experiment_name": str(mlflow_meta["experiment_name"]),
            "parent_run_id": str(mlflow_meta["parent_run_id"]),
            "parent_run_name": str(mlflow_meta["parent_run_name"]),
        }
        MlflowClient(tracking_uri=context["tracking_uri"]).set_tag(
            context["parent_run_id"], "orchestrator_status", "running"
        )
        return context

    tracking_uri = resolve_mlflow_tracking_uri(cfg.get("mlflow_tracking_uri"))
    experiment_name = _ablation_experiment_name(cfg)
    created_at = datetime.now(timezone.utc)
    run_stamp = created_at.strftime("%Y%m%d_%H%M%SZ")
    phase_token = phase.lower()
    parent_run_name = (
        f"{cfg['variant']}__{cfg['split_name']}__{phase_token}__{run_stamp}"
    )

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    parent_run = client.create_run(
        experiment_id=experiment.experiment_id,
        run_name=parent_run_name,
        tags={
            "run_kind": "ablation_root",
            "orchestrator": "scripts/run_ablation.py",
            "dataset": cfg["dataset"],
            "fold": cfg["split_name"],
            "variant": cfg["variant"],
            "phase_filter": phase,
            "config_path": str(config_path),
            "state_file": str(_state_path(run_id)),
            "orchestrator_status": "running",
        },
    )

    with mlflow.start_run(run_id=parent_run.info.run_id):
        mlflow.log_params(
            {
                "config_path": str(config_path),
                "dataset": cfg["dataset"],
                "variant": cfg["variant"],
                "fold": cfg["split_name"],
                "phase_filter": phase,
                "state_file": str(_state_path(run_id)),
            }
        )

    mlflow_meta.update(
        {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "parent_run_id": parent_run.info.run_id,
            "parent_run_name": parent_run_name,
            "created_at": created_at.isoformat(),
        }
    )
    save_state(run_id, state)
    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "parent_run_id": parent_run.info.run_id,
        "parent_run_name": parent_run_name,
    }


def _build_step_env(
    *,
    mlflow_context: dict[str, str],
    config_path: Path,
    run_id: str,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            ABLATION_TRACKING_URI_ENV: mlflow_context["tracking_uri"],
            ABLATION_EXPERIMENT_ENV: mlflow_context["experiment_name"],
            ABLATION_PARENT_RUN_ENV: mlflow_context["parent_run_id"],
            ABLATION_ROOT_RUN_ENV: mlflow_context["parent_run_id"],
            ABLATION_RUN_KEY_ENV: run_id,
            ABLATION_CONFIG_ENV: str(config_path),
        }
    )
    return env


def _update_root_run_progress(
    mlflow_context: dict[str, str],
    *,
    state: dict[str, Any],
    total_steps: int,
) -> None:
    completed_steps = sum(
        1
        for key, value in state.items()
        if not str(key).startswith("_")
        and isinstance(value, dict)
        and value.get("status") == "done"
    )
    mlflow.set_tracking_uri(mlflow_context["tracking_uri"])
    with mlflow.start_run(run_id=mlflow_context["parent_run_id"]):
        mlflow.log_metric("steps_completed", float(completed_steps))
        mlflow.log_metric("steps_total", float(total_steps))


def _set_root_run_status(
    mlflow_context: dict[str, str],
    *,
    status_tag: str,
    error_message: str | None = None,
    terminate_status: str | None = None,
) -> None:
    client = MlflowClient(tracking_uri=mlflow_context["tracking_uri"])
    client.set_tag(mlflow_context["parent_run_id"], "orchestrator_status", status_tag)
    if error_message:
        client.set_tag(
            mlflow_context["parent_run_id"],
            "orchestrator_error",
            error_message[:250],
        )
    if terminate_status:
        client.set_terminated(
            mlflow_context["parent_run_id"],
            status=terminate_status,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

_BANNER = "=" * 72


def _print_step_header(step: dict[str, Any], index: int, total: int) -> None:
    logger.info(_BANNER)
    logger.info(
        "Step %d/%d  [Phase %s]  %s", index + 1, total, step["phase"], step["name"]
    )
    logger.info("ID: %s", step["id"])


def _resolve_step_cmd(step: dict[str, Any]) -> str:
    """
    Resolve any ``{placeholder}`` tokens in a step's cmd that are listed under
    the ``resolve`` key.  Each resolver of type ``mlflow_artifact`` looks up the
    most-recent MLflow run of the given experiment and returns its artifact dir.
    """
    cmd: str = step["cmd"]
    resolvers: dict[str, Any] = step.get("resolve", {})
    if not resolvers:
        return cmd

    substitutions: dict[str, str] = {}
    for placeholder, spec in resolvers.items():
        if spec.get("type") == "mlflow_artifact":
            value = _find_mlflow_checkpoint(spec["mlflow_uri"], spec["experiment"])
        else:
            value = f"<unknown resolver type '{spec.get('type')}'>"
        substitutions[placeholder] = value
        logger.info("  Resolved %s → %s", placeholder, value)

    try:
        return cmd.format_map(substitutions)
    except KeyError as exc:
        logger.warning("Could not resolve placeholder %s in cmd — leaving as-is.", exc)
        return cmd


def run_step(
    step: dict[str, Any],
    run_id: str,
    state: dict[str, Any],
    dry_run: bool,
    step_env: dict[str, str],
) -> bool:
    """
    Execute a single step.

    Returns True  → continue to next step.
    Returns False → pipeline paused (wait step reached).
    """
    step_id: str = step["id"]
    raw_cmd: str = step["cmd"]

    # Already completed?
    if state.get(step_id, {}).get("status") == "done":
        logger.info("  ✓ SKIP (already completed)")
        return True

    # Previously paused here?
    if state.get(step_id, {}).get("status") == "waiting":
        logger.info("  … Resuming past wait step — marking done and continuing.")
        mark_done(run_id, state, step_id, raw_cmd)
        return True

    # Resolve any {placeholder} tokens before executing.
    cmd = _resolve_step_cmd(step)

    if dry_run:
        logger.info("  DRY-RUN cmd:\n    %s", cmd.replace("  ", " ").strip())
        return True

    # Training steps are highlighted in the plan, but the pipeline keeps going
    # automatically once the command finishes.
    if step.get("wait"):
        logger.info(_BANNER)
        logger.info("⚡ TRAINING STEP — launching and waiting for it to finish.")
        logger.info("   The pipeline will continue automatically after it completes.")
        logger.info(_BANNER)
        logger.info("  cmd: %s", cmd.replace("  ", " ").strip())
        result = subprocess.run(cmd, shell=True, env=step_env)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        mark_done(run_id, state, step_id, cmd)
        logger.info("  ✓ Training step finished")
        return True

    logger.info("  cmd: %s", cmd.replace("  ", " ").strip())
    result = subprocess.run(cmd, shell=True, env=step_env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)

    mark_done(run_id, state, step_id, cmd)
    logger.info("  ✓ Done")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to variant YAML (e.g. experiments/variants/baseline.yaml).",
)
@click.option(
    "--fold",
    "-f",
    required=True,
    type=click.Choice(["1", "2", "mixed"], case_sensitive=False),
    help="Split to run: 1, 2, or mixed.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print resolved commands without executing anything.",
)
@click.option(
    "--phase",
    type=click.Choice(["A", "B", "C", "all"]),
    default="all",
    show_default=True,
    help="Run only steps belonging to the specified phase.",
)
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="Clear checkpoint state and re-run all steps from scratch.",
)
@click.option(
    "--list-steps",
    is_flag=True,
    default=False,
    help="Print all steps that would be run (with IDs) and exit.",
)
def main(
    config: Path,
    fold: str,
    dry_run: bool,
    phase: str,
    reset: bool,
    list_steps: bool,
) -> None:
    """
    Run the ablation pipeline for one fold.

    Steps are checkpointed in .state/<run_id>.json — re-running after a failure
    resumes from the last completed step.
    """
    cfg = load_config(config, fold)
    variant = cfg["variant"]
    dataset = cfg["dataset"]
    run_id = f"{dataset}_{variant}_{cfg['split_name']}"

    steps = build_steps(cfg)

    # Phase filter
    if phase != "all":
        steps = [s for s in steps if s["phase"] == phase]

    if list_steps:
        click.echo(f"\nRun ID: {run_id}\n")
        for i, s in enumerate(steps):
            click.echo(
                f"  [{s['phase']}] {i + 1:02d}. {s['id']}\n"
                f"        {s['name']}" + (" ⏱ TRAINING" if s.get("wait") else "")
            )
            if s.get("output_hint"):
                click.echo(f"        → output: {s['output_hint']}")
            if s.get("resolve"):
                for ph, spec in s["resolve"].items():
                    click.echo(
                        f"        → resolves {{{ph}}} from MLflow experiment: {spec.get('experiment','?')}"
                    )
        click.echo(f"\n{len(steps)} steps total.\n")
        return

    state_path = _state_path(run_id)
    if reset and state_path.exists():
        state_path.unlink()
        logger.info("Cleared checkpoint state for run '%s'.", run_id)

    state = load_state(run_id)
    mlflow_context: dict[str, str] | None = None
    step_env = dict(os.environ)

    if not dry_run:
        mlflow_context = _build_mlflow_context(cfg, config, run_id, phase, state)
        step_env = _build_step_env(
            mlflow_context=mlflow_context,
            config_path=config,
            run_id=run_id,
        )
        logger.info(
            "MLflow experiment: %s   Parent run: %s",
            mlflow_context["experiment_name"],
            mlflow_context["parent_run_name"],
        )
        _update_root_run_progress(
            mlflow_context,
            state=state,
            total_steps=len(steps),
        )

    logger.info(_BANNER)
    logger.info("Ablation pipeline — run ID: %s", run_id)
    logger.info(
        "Config: %s   Split: %s   Phase filter: %s",
        config,
        cfg["split_name"],
        phase,
    )
    logger.info("State file: %s", _state_path(run_id))
    if dry_run:
        logger.info("MODE: DRY-RUN — no commands will be executed")
    logger.info(_BANNER)

    try:
        for i, step in enumerate(steps):
            _print_step_header(step, i, len(steps))
            run_step(step, run_id, state, dry_run, step_env)
            if mlflow_context is not None:
                _update_root_run_progress(
                    mlflow_context,
                    state=state,
                    total_steps=len(steps),
                )
    except subprocess.CalledProcessError as exc:
        if mlflow_context is not None:
            _set_root_run_status(
                mlflow_context,
                status_tag="failed",
                error_message=f"{exc.cmd} exited with code {exc.returncode}",
            )
        logger.error(
            "  ✗ Step failed (exit %d). Fix and re-run.",
            exc.returncode,
        )
        sys.exit(exc.returncode)

    logger.info(_BANNER)
    logger.info("✅ All steps complete for run '%s'.", run_id)
    logger.info(_BANNER)
    if mlflow_context is not None:
        _set_root_run_status(
            mlflow_context,
            status_tag="finished",
            terminate_status="FINISHED",
        )


if __name__ == "__main__":
    main()
