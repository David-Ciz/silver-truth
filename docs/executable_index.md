# Executable Index

This page is a task-oriented map of what agents should check before adding new code.

## Start Here

Prefer these surfaces in this order:

1. Console entrypoints from `pyproject.toml`
2. Maintained orchestration scripts in `scripts/`
3. Reusable functions in `src/silver_truth/...`
4. Only then legacy scratch files or notebooks

If a task is already covered by a CLI command or a tested function, reuse that path instead of creating another script.

## Console Entrypoints

These are the supported package commands defined in `pyproject.toml`:

- `silver-preprocessing`
  - dataset synchronization, dataframe creation, TIFF compression, GT/cell-size statistics
- `silver-qa`
  - QA crop generation, split attachment, QA CNN train/eval helpers
- `silver-fusion`
  - job-file generation, Java fusion, crop-level fusion experiments
- `silver-evaluation`
  - competitor evaluation, fusion reconstruction scoring, QA evaluation, parquet filtering
- `silver-ensemble`
  - ensemble databank build, training, checkpoint evaluation

For the current command map, see [Repository Workflow Map](repository_workflow_map.md) and [Function Index](function_index.md).

## Maintained Scripts

These are the repository-level scripts that currently have clear documentation support and should be preferred over ad hoc files:

- `scripts/run_ablation.py`
  - maintained paper runner; orchestrates Phase A/B/C from YAML configs in `experiments/variants/`
  - supports `--workflow reduced` for a smaller first pass and `--workflow full` to expand later without `--reset`
- `scripts/run_ablation_hpc.sh`
  - maintained Slurm wrapper for the ablation runner; stages data to scratch and isolates outputs under a durable campaign root
  - forwards `--workflow {config|reduced|full}` and records the workflow in durable logs / job index
- `scripts/submit_hela_ablation_hpc.sh`
  - convenience submitter for the first `DIC-C2DH-HeLa` `sz256` ablation jobs; runs dataset readiness checks before Slurm submission and defaults logs to `/home/davidciz/silver-truth/logs`
- `scripts/check_dataset_readiness.py`
  - pre-HPC gate for dataset preprocessing, DVC artifact presence, crop-size fit, split sanity, and QA crop TIFF dimensions
- `scripts/run_qa_transfer.py`
  - maintained zero-shot QA transfer runner; evaluates a source QA checkpoint on a different dataset/split and writes transfer-specific artifacts under `paper_runs/qa_transfer/`
- `scripts/run_qa_transfer_hpc.sh`
  - Slurm wrapper for the QA transfer runner; stages source/target datasets to scratch and logs a dedicated transfer run bundle
- `scripts/analyze_dataset_profiles.py`
  - compares synchronized datasets at the raw-frame and GT-object level; writes crop-fit and size-profile reports under `data/analysis/dataset_profiles/`
- `scripts/analyze_qa_prediction_bundle.py`
  - audits saved QA prediction Excel bundles for split drift, timepoint-level shift, and residual calibration by score range; useful for follow-up analysis after ablation runs
- `scripts/generate_api_docs.py`
  - refreshes `docs/api_reference_generated.md` from source docstrings

For a human-oriented explanation of the ablation runner itself, see [Ablation Runner Explained](ablation_runner_explained.md).

Other files in `scripts/` may still be useful, but many are experiment-specific helpers. Before reusing one, verify that it is still referenced by the docs or current workflows.

## Legacy / Ad Hoc Surfaces

These exist, but should not be the default reuse path:

- `ztest_commands.py`
  - historical scratch workflow; not part of the maintained CLI
- `notebooks/`
  - exploratory analysis and one-off debugging
- `scripts/analyze_segmentation.py`
  - visualization helper for a single TIFF mask; not the maintained way to compute dataset-wide size statistics

## Task Lookup

### Find GT Cell Sizes Across a Dataset

Use this maintained command with either GT directories or whole-image dataset parquets:

```bash
silver-preprocessing segmentation-size-stats \
  data/dataframes/BF-C2DL-MuSC/whole_image/BF-C2DL-MuSC_split_mixed.parquet \
  --crop-size 64 --crop-size 96 --crop-size 128 \
  --rect-size 128x256
```

Supported source functions:

- `silver_truth.data_processing.segmentation_stats.collect_segmentation_object_stats`
  - scans labeled GT masks and records per-cell area, bbox height/width, and aspect-ratio fields
- `silver_truth.data_processing.segmentation_stats.collect_segmentation_object_stats_from_dataframes`
  - reads whole-image dataset parquets and resolves GT and source-image context from parquet metadata
- `silver_truth.data_processing.segmentation_stats.summarize_segmentation_object_stats`
  - computes overall and per-directory summary stats, square/rectangular fit rates, and bbox aspect-ratio bias summaries
- `silver_truth.cli.preprocessing.segmentation_size_stats`
  - CLI wrapper that prints JSON and can also save summary/per-object outputs and optional outlier audits

Useful options:

- `--output <path>.json`
  - save the aggregated summary
- `--per-object-output <path>.parquet`
  - save one row per GT cell for follow-up analysis

Interpretation:

- `fit_bbox_rate_sz128` means the fraction of GT cells whose bbox height and width are both `<= 128`
- `over_bbox_count_sz128` means how many GT cells would still be too large for a `128x128` crop
- `fit_bbox_rate_rect128x256` means the fraction of GT cells that fit a fixed axis-aligned `128x256` crop
- `fit_bbox_rate_rect128x256_swappable` means the fraction of GT cells that fit either `128x256` or `256x128`
- `max_bbox_dim_px` is the single largest bbox dimension seen anywhere in the input folders
- `mean_bbox_aspect_ratio_wh` is the average `width / height` ratio
- `wide_bbox_rate` and `tall_bbox_rate` show whether the dataset is systematically wider or taller
- `mean_bbox_elongation_ratio` shows how stretched boxes are regardless of direction

For MuSC specifically, use the current crop-size rerun path in
[Clean Slate Experiment Rerun Plan](clean_slate_experiment_rerun_plan_2026-05-05.md)
and [Rerun Hypotheses Backlog](rerun_hypotheses_backlog_2026-05-05.md).

### Check Dataset Readiness Before HPC

Use the readiness gate before launching reduced or full ablations:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_dataset_readiness.py \
  --config experiments/variants/hela_crop256.yaml \
  --fold 1 \
  --fold 2
```

This writes JSON/Markdown reports under `data/audits/dataset_readiness/` and
prints the exact `dvc repro ...` command if split-aware parquets or QA crop
artifacts are missing. See [Dataset Readiness Checks](dataset_readiness_checks.md).

### Audit Cells That Would Be Clipped By a Chosen Square Crop

Use the same command with `--show-outliers-for` when you want parquet-driven inspection with both GT and source-image context:

```bash
silver-preprocessing segmentation-size-stats \
  data/dataframes/BF-C2DL-MuSC/whole_image/BF-C2DL-MuSC_split_mixed.parquet \
  --crop-size 128 --crop-size 256 --crop-size 512 \
  --show-outliers-for 256 \
  --outlier-output-dir data/audits/musc_sz256
```

This mode:

- reads `gt_image` and `source_image` from the dataset parquet
- finds GT cells whose bbox height or width exceeds the requested square crop size
- writes a CSV audit table
- writes raw+GT context PNGs so you can inspect likely merged/pathological cases

### Check Whether Overflow Cells Actually Change Metrics

When you already have a per-cell result table, use:

```bash
silver-evaluation report-overflow-impact \
  results/BF-C2DL-MuSC_fold-2_detailed.parquet \
  data/dataframes/BF-C2DL-MuSC/whole_image/BF-C2DL-MuSC_split_fold-2.parquet \
  --crop-size 256 \
  --output-dir data/audits/musc_fold2_overflow_impact
```

This command:

- reads a per-cell result parquet/CSV such as `evaluate-competitor --detailed`
- computes which GT cells overflow the chosen crop size
- writes enriched per-cell results plus summary tables split into `fits` vs `overflow`

### If You Encounter the Older Largest-Cell Helper

There is also an older helper:

- `silver_truth.ensemble.utils.find_largest_gt_cell_size`

That helper:

- reads GT image paths from a parquet
- returns only the single largest bbox dimension and source image
- is currently referenced from `ztest_commands.py`

Use `segmentation-size-stats` instead when the real question is dataset-wide crop-size safety or minimum usable box size.

### Compare Multiple Datasets Before Opening a New Crop-Size Branch

Use:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_dataset_profiles.py
```

This script:

- scans `DIC-C2DH-HeLa`, `BF-C2DL-HSC`, and `BF-C2DL-MuSC` by default
- records raw-frame counts, GT-frame counts, competitor counts, and image sizes
- computes GT bbox fit rates across a grid of crop sizes
- writes reusable outputs under `data/analysis/dataset_profiles/`

Current reference note:

- [HeLa, HSC, and MuSC Profile Notes](datasets/hela_hsc_musc_profile.md)

### Audit Why a QA Fold Behaves Differently Across Train / Validation / Test

Use:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_qa_prediction_bundle.py \
  --input-root data/paper_runs/qa_results/<DATASET>/<CROP_TAG>/<VARIANT> \
  --output-dir data/analysis/qa_prediction_bundle/<DATASET>/<CROP_TAG>/<VARIANT>
```

This script:

- scans one or more `*_qa_predictions_*.xlsx` files
- summarizes split composition, including campaign and timepoint coverage
- reports per-timepoint true/predicted quality and residual bias
- bins residuals by true Jaccard range so calibration failures are easy to spot

Current reference note:

- [Rerun Hypotheses Backlog](rerun_hypotheses_backlog_2026-05-05.md)

The old HeLa result notes were archived under
`archive/docs_pre_split_fix_2026-05-05/` and should not be used as result
sources.
