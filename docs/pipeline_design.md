# Ablation Pipeline Design
_Written: 2026-03-09_

This page is now a design-to-implementation bridge. It has been updated to reflect the current repository state rather than the original proposal.

A lightweight, layered system built on top of what already exists.
No new frameworks — YAML configs, a thin Python runner, consistent MLflow tags,
and a results aggregation script.

Total new code: ~500 lines across 5 files.

---

## Overview

```
experiments/
  base.yaml                     # canonical defaults
  variants/
    baseline.yaml               # no overrides (extends base)
    resnet18_qa.yaml            # qa_model_type: resnet18
    unet_ensemble.yaml          # ensemble_model_type: Unet
    fluo_dataset.yaml           # dataset: Fluo-N2DL-HeLa
    crop32.yaml                 # crop_size: 32
scripts/
  run_ablation.py               # pipeline runner — reads variant YAML, sequences CLI calls
```

---

## Piece 1 — Implemented CLI Surfaces

### `silver-evaluation evaluate-fusion-crops`
File: `src/silver_truth/cli/evaluation.py`

Calls `reconstruct_full_images_from_paths` from `ensemble/reconstruction.py`.

**Column name mapping** (confirmed): `run-fusion-crops` output parquet retains the
original QA parquet columns (`crop_y_start/end/x_start/x_end`) and adds `{model_lower}`
(fused path) and `crop_gt_path`. The `reconstruct_full_images_from_paths` function expects
`recon_crop_y_start/end/x_start/x_end`. The CLI command renames on the fly before passing
to the reconstruction function — no in-place mutation of the parquet.

```bash
silver-evaluation evaluate-fusion-crops \
  data/paper_runs/fusion/fold-1/bic_flat_voting/fold-1_sz64_qa_dataset_bic_flat_voting_with_fused.parquet \
  --fused-path-column bic_flat_voting \
  --output-dir data/paper_runs/fusion/fold-1/bic_flat_voting_fullimage/ \
  --output data/paper_runs/fusion/fold-1/bic_flat_voting_fullimage_eval.csv
```

### `silver-evaluation filter-parquet`
File: `src/silver_truth/cli/evaluation.py`

Thin pandas wrapper around two filtering modes. Used as a preprocessing step before
passing to `run-fusion-crops` or `evaluate-fusion-crops`.

```bash
# qa_only — top-1 predicted_jaccard_index per cell, no fusion
silver-evaluation filter-parquet \
  data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_paper_ready.parquet \
  --mode qa_only \
  --output data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_qa_only.parquet

# full_pipeline — threshold + fallback top-1 for unrepresented cells
silver-evaluation filter-parquet \
  data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_paper_ready.parquet \
  --mode full_pipeline \
  --threshold 0.75 \
  --output data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_full_pipeline_t0.75.parquet
```

---

## Piece 2 — Experiment Variant YAML Configs

File: `experiments/base.yaml`

```yaml
# Base experiment configuration — all variants inherit these defaults.
dataset: BF-C2DL-HSC
folds: [1, 2]
crop_size: 64

# Models
qa_model_type: resnet50
qa_input_channels: "0,1"
ensemble_model_type: UnetPlusPlus
ensemble_max_epochs: 100
fusion_models: [BIC_FLAT_VOTING]

# QA filtering
qa_threshold: 0.75

# Paths
mlflow_tracking_uri: "file:{project_root}/data/mlflow/mlruns"
data_root: "data/dataframes/{dataset}"
paper_runs_root: "data/paper_runs"

# Ablation modes to run in Phase C
ablation_modes:
  - fusion_only
  - qa_only
  - full_pipeline
  - ensemble_only
  - ensemble_qa
  - ensemble_qa_retrained
```

Variant files override only what changes:

```yaml
# experiments/variants/resnet18_qa.yaml
extends: base
qa_model_type: resnet18
```

```yaml
# experiments/variants/crop32.yaml
extends: base
crop_size: 32
```

---

## Piece 3 — Pipeline Runner

File: `scripts/run_ablation.py`

### Key design decisions

- **YAML merging**: reads `base.yaml`, overlays variant file. All paths are template strings
  resolved with `str.format_map(config)`.
- **Checkpoint file**: `.state/{run_id}.json` written after each step success with
  step status/timestamp/command. Re-running skips completed steps.
- **Parallelism**: folds are independent — launch two processes with `--fold 1` and `--fold 2`.
- **Training behavior**: training steps run inline and the pipeline continues automatically
  after the subprocess finishes. Re-run only if the shell was interrupted.
- **Dry run**: `--dry-run` prints the resolved command list without executing.

### Usage

```bash
# Run baseline (all phases, fold 1)
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1

# Dry-run to inspect commands before executing
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1 --dry-run

# Resume after QA training completed
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1
# (automatically skips completed steps)

# Test a different QA model
python scripts/run_ablation.py --config experiments/variants/resnet18_qa.yaml --fold 1

# Test on a different dataset (add experiments/variants/fluo_dataset.yaml first)
python scripts/run_ablation.py --config experiments/variants/fluo_dataset.yaml --fold 1
```

---

## Piece 4 — MLflow Tagging

File: `src/silver_truth/experiment_tracking.py`

Add `set_ablation_tags(run, dataset, fold, phase, step, variant, git_sha)`.

MLflow experiment naming convention: `{phase}-{dataset}-{variant}-{fold}`

Examples:
- `phaseA-BF-C2DL-HSC-baseline-fold1`
- `phaseC-BF-C2DL-HSC-resnet18_qa-fold2`

Tags on every run: `dataset`, `fold`, `variant`, `ablation_mode`, `qa_threshold`, `git_sha`.

This makes the MLflow Compare Runs UI filterable by any axis without restructuring.

---

## Piece 5 — Results Aggregation (Still Missing)

There is currently no maintained `scripts/collect_paper_results.py` in the repository.

Current result surfaces are:
- per-step CSV/parquet artifacts in `data/paper_runs/`
- MLflow metrics and artifacts
- `.state/*.json` runner checkpoints

The missing piece is a final paper-reporting layer that would consolidate:
- cross-fold summary tables
- bootstrap confidence intervals
- paired significance tests

---

## Build Order

1. ✅ `evaluate-fusion-crops` and `filter-parquet` are implemented
2. ✅ `experiments/base.yaml` + variant files are in place
3. ✅ `scripts/run_ablation.py` sequences Phase A/B/C runs
4. ✅ MLflow tagging helper exists
5. 🔜 Final paper-reporting aggregation remains to be implemented

---

## Further Considerations

### Column name mismatch confirmed
`run-fusion-crops` output uses `crop_y_start/end/x_start/x_end` (from the QA parquet);
`reconstruct_full_images_from_paths` expects `recon_crop_*`. The `evaluate-fusion-crops`
command handles the rename. Nothing else reads `_with_fused.parquet` and relies on
`recon_crop_*` names, so no in-place changes to the parquet schema are needed.

### DVC for deterministic steps
Non-deterministic training (QA, ensemble) stays MLflow-only.
Deterministic post-training steps (filter-parquet, run-fusion-crops, evaluate-fusion-crops,
evaluate-competitor) can become DVC stages parameterized via `params.yaml`
(`ablation.qa_threshold`, `ablation.fusion_model`) for `dvc repro` reproducibility of the
final paper numbers.

### Multi-dataset / crop-size variants
`params.yaml` currently includes both `BF-C2DL-HSC` and `BF-C2DL-MuSC`. Adding a second dataset or
crop size means adding a new `experiments/variants/` YAML — the runner handles the rest.
