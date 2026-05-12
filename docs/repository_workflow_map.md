# Repository Workflow Map

This page documents how this repository currently works end-to-end, based on the actual CLI code, `dvc.yaml`, and the current README/docs.

## Scope and Source of Truth

- CLI behavior is defined in `src/silver_truth/cli/*.py` and their called modules.
- Deterministic data preparation is defined in `dvc.yaml`.
- Paper experiment orchestration is implemented in `scripts/run_ablation.py` with YAML configs in `experiments/`.

## Executable Surfaces

The package defines these console entrypoints (in `pyproject.toml`):

- `silver-preprocessing`
- `silver-qa`
- `silver-fusion`
- `silver-evaluation`
- `silver-ensemble`

Note: these are only available after environment setup (`pip install -e .[dev]`).

## CLI Command Map

### `silver-preprocessing`

- `verify-dataset-synchronization`
- `verify-folder-synchronization`
- `synchronize-labels`
- `synchronize-datasets`
- `create-dataset-dataframe`
- `segmentation-size-stats`
- `compress-tifs`

Core behavior:

- Synchronizes label masks with tracking markers.
- Builds whole-image parquet dataframes from synchronized dataset folders.
- Reports GT cell area, bbox size, aspect-ratio bias, and candidate crop-size fit rates from segmentation folders or dataset parquets.
- Can optionally audit oversized GT cells from dataset parquets and generate source/GT context visualizations for crop-size review.
- Assigns splits with either:
- `mixed` strategy (`train/validation/test` via GT-cell-count balancing).
- `fold-1` or `fold-2` strategy (leave-one-sequence-out, with validation
  selected inside the held-in sequence by supervised GT-cell support).
- Stores competitor columns in parquet metadata (`df.attrs["competitor_columns"]`).

### `silver-qa`

- `create-dataset`
- `attach-split` (docstring says DEPRECATED, but used in DVC)
- `evaluate`
- `convert-results`
- `batch-convert-results`
- `cnn train`
- `cnn evaluate`

Core behavior:

- Creates QA crops (or full-image QA stacks) from whole-image dataframe rows with valid GT.
- Crop TIFF layout for `--crop` mode is `(4, H, W)`:
- channel 0 raw image
- channel 1 competitor mask (binary)
- channel 2 GT mask (binary)
- channel 3 tracking marker mask (binary)
- Supports centering strategies: `competitor_consensus`, `competitor_individual`, `gt_mask`, `tracking_marker`.
- `attach-split` joins base QA parquet to whole-image split parquet (preferred keys: `campaign_number + gt_image`).

### `silver-fusion`

- `run-fusion`
- `generate-jobfiles`
- `add-fused-images`
- `run-fusion-crops`

Core behavior:

- `run-fusion` is a Python wrapper around Java `RunFusersCli` using `FusionModel` enum modes.
- `generate-jobfiles` builds one job file per dataset/campaign from parquet competitor columns and tracking marker paths.
- `add-fused-images` maps produced fused TIFFs back into parquet columns.
- `run-fusion-crops` is the MLflow-tracked crop experiment orchestrator:
- stages QA rows into synthetic timepoints
- runs one or more fusion models (with optional chunking/fallback)
- evaluates cell-level IoU/F1
- writes model summary, leaderboard, and enriched parquet artifacts
- logs metrics/artifacts/tags to MLflow

### `silver-evaluation`

- `evaluate-competitor`
- `calculate-evaluation-metrics-cli`
- `evaluate-qa-model`
- `evaluate-qa-filtering`
- `merge-qa-predictions`
- `evaluate-fusion-crops`
- `filter-parquet`

Core behavior:

- Computes crop/full evaluation metrics and writes back to parquet.
- Evaluates QA model predictions from Excel and can log to MLflow.
- Merges QA prediction columns (e.g., `predicted_jaccard_index`) into parquet.
- Reconstructs fused crops back to full images for paper-level IoU/F1 scoring.
- Filters QA-enriched parquets for `qa_only` and thresholded `full_pipeline` modes.

### `silver-ensemble`

- `ensemble-experiment`
- `build-databank`
- `evaluate-checkpoint`
- `evaluate-best-checkpoint`

Core behavior:

- Builds ensemble databank parquet/assets from QA parquet.
- Trains/evaluates ensemble checkpoints.

## Data and Artifact Flow

High-level path:

1. Synchronized dataset folder
2. Whole-image dataframe parquet (`create-dataset-dataframe`)
3. Base QA crops + parquet (`silver-qa create-dataset --crop`)
4. Split-specific QA parquet (`silver-qa attach-split`)
5. QA labels for crops (`silver-evaluation calculate-evaluation-metrics-cli --mode cropped`)
6. QA model predictions merged (`merge-qa-predictions`)
7. Experiment orchestration via `scripts/run_ablation.py` or direct manual CLI calls
8. Per-step CSV/parquet artifacts under `data/paper_runs/`, plus MLflow runs and `.state/` runner checkpoints

Key directories used by current workflows:

- `data/synchronized_data/...`
- `data/dataframes/{dataset}/whole_image/...`
- `data/qa_crops/{dataset}/sz{crop_size}/...`
- `data/dataframes/{dataset}/qa_crops/...`
- `data/job_files/{dataset}/{split}/...`
- `data/paper_runs/...`
- `data/mlflow/mlruns`

## What `dvc.yaml` Currently Automates

Defined stages include:

- `create_fold1` (for `BF-C2DL-HSC`, `BF-C2DL-MuSC`, `DIC-C2DH-HeLa`)
- `create_fold2` (for `BF-C2DL-HSC`, `BF-C2DL-MuSC`, `DIC-C2DH-HeLa`)
- `create_mixed` (for `BF-C2DL-HSC`, `BF-C2DL-MuSC`, `DIC-C2DH-HeLa`)
- `create_qa_crops_base` (foreach `${datasets}`)
- `create_qa_crops_split_mixed` (foreach `${datasets}`)
- `create_qa_crops_split_fold1` (foreach `${datasets}`)
- `create_qa_crops_split_fold2` (foreach `${datasets}`)
- `create_ensemble_databank_c1_hsc_mixed`
- `generate_job_files_mixed` (HSC and HeLa campaigns 01/02)
- `generate_job_files_fold1` (HSC and HeLa campaigns 01/02)
- `generate_job_files_fold2` (HSC and HeLa campaigns 01/02)

Important:

- DVC currently covers data preparation and job-file generation.
- DVC does not run the paper experiment runner or QA CNN training.
- Fusion execution/evaluation is primarily handled as MLflow experiments outside DVC.

## Paper Ablation Mechanics (Actual Implementation)

The maintained implementation path is `scripts/run_ablation.py`, which wires together:

- `silver-evaluation evaluate-competitor`
- `silver-qa cnn train`
- `silver-evaluation evaluate-qa-model`
- `silver-evaluation evaluate-qa-filtering`
- `silver-evaluation merge-qa-predictions`
- `silver-evaluation filter-parquet`
- `silver-fusion run-fusion-crops`
- `silver-evaluation evaluate-fusion-crops`
- `silver-ensemble build-databank`
- `silver-ensemble ensemble-experiment`
- `silver-ensemble evaluate-best-checkpoint`

The fold protocol is encoded by config-selected input paths and split naming:

- `fold-1` test campaign must be `02`; non-test must be `01`.
- `fold-2` test campaign must be `01`; non-test must be `02`.

Modes:

- `fusion_only`: no QA gate, all candidates fused with Java model.
- `full_pipeline`: threshold gate on QA score, then Java fusion.
- `qa_only`: threshold-filtered top-1 QA candidate per cell, no voting fusion.
- `ensemble_only`: evaluate the unfiltered trained ensemble.
- `ensemble_qa`: evaluate the trained ensemble on a QA-filtered databank.
- `ensemble_qa_retrained`: retrain the ensemble on QA-filtered input, then evaluate.

Selection safeguard:

- If a cell has no candidate above threshold, deterministic top-1 fallback is used by `silver-evaluation filter-parquet`.

Metrics:

- Cell-level IoU/F1 is computed on staged masks.
- Full-image primary metrics are reconstructed from fused crops with sparse-GT label-aware scoring.
- There is not yet a single final paper-reporting aggregator for all folds/modes.
- Canonical outputs today are the per-step CSV/parquet artifacts in `data/paper_runs/`, MLflow metrics, and `.state/*.json`.

## Known Operational Notes

- The fusion command used by the maintained experiment path is `silver-fusion run-fusion-crops`.
- `silver-qa attach-split` is marked DEPRECATED in docstring but is still used in DVC stages.
- `dvc.yaml` and `dvc.lock` can diverge during active development; trust current `dvc.yaml` for intended pipeline definition.
- `params.yaml` currently defines `BF-C2DL-HSC`, `BF-C2DL-MuSC`, and `DIC-C2DH-HeLa` under `datasets`.
- The first committed HeLa crop baseline is `sz256`, based on the direct GT bbox comparison captured in [HeLa, HSC, and MuSC Profile Notes](datasets/hela_hsc_musc_profile.md).

## Practical Run Order (Current Recommended)

For fold-locked paper runs:

1. Follow [Paper Protocol](paper_protocol.md) and the
   [Clean Slate Experiment Rerun Plan](clean_slate_experiment_rerun_plan_2026-05-05.md).
2. Reproduce fold and QA split stages via DVC.
3. Run preflight audits before training.
4. Run `python scripts/run_ablation.py --config <variant.yaml> --fold <1|2>`.
5. Record current artifacts in [Manuscript Artifact Registry](manuscript_artifact_registry.md).

For fusion baseline comparison on QA crops:

1. Prepare split QA parquet.
2. Run `silver-fusion run-fusion-crops`.
3. Compare `fusion_crops_summary.csv`, `fusion_crops_leaderboard.csv`, and MLflow model runs.
