# MLflow Ablation Rewrite Plan
_Drafted: 2026-03-12_

This plan captures the MLflow and evaluation rewrite needed after reviewing the BF-C2DL-HSC ablation runs in the MLflow UI.

---

## Problem Statement

The current ablation run logs are hard to compare because they mix training telemetry with final evaluation, reuse different metric names for the same idea, and often do not say clearly whether a number is:

- label-level on full images
- crop/cell-level
- reconstructed full-image evaluation
- QA regression or QA thresholding metadata

The result is that the MLflow UI does not currently answer the main paper question cleanly:

> "How do the different setups compare on `test_iou` and `test_f1`, and at what evaluation level were those numbers produced?"

---

## Direct Findings From UI Review

### 1. Naming is inconsistent

- Experiment names do not always say what the run family is.
- Parent run names are sometimes timestamp-like or randomly generated instead of descriptive.
- Nested run structure is inconsistent across competitors, fusion, ensemble, QA, and QA filtering.

### 2. Final evaluation metrics are not standardized

The same concept is logged under different names:

- `jaccard_test`
- `test_mean_jaccard`
- `iou_mean`
- `val_iou`, `val_f1`

For paper comparison, every final-evaluation run should expose:

- `test_iou`
- `test_f1`
- `val_iou`
- `val_f1`
- `train_iou`
- `train_f1`
- `overall_iou`
- `overall_f1`

Legacy names can stay for backward compatibility, but the standard aliases need to exist everywhere.

### 3. Evaluation level is not obvious

Every comparable run must say, front and center, what was evaluated:

- `evaluation_level=full_image_label`
- `evaluation_level=image_reconstructed`
- `evaluation_level=cell_crop`
- `evaluation_level=qa_regression`
- `evaluation_level=qa_threshold_filtering`

### 4. Ensemble final evaluation is underrepresented

The MLflow UI currently shows ensemble training metrics clearly enough, but the comparable final evaluation is missing or not obvious:

- unfiltered ensemble inference on test
- ensemble inference on QA-filtered inputs
- retrained ensemble on QA-filtered inputs

These need explicit final-evaluation runs with the same canonical metric names.

### 5. QA filtering experiment is semantically unclear

`phaseB-qa-filtering-*` currently logs thresholded classification metrics over QA predictions. That is valid, but it is not the same as:

- selecting the best competitor crop per cell
- evaluating that selector as a final segmentation setup

Those are two different experiments and should be logged separately.

### 6. Threshold exploration is missing

We need automated threshold sweeps for:

- fusion after QA filtering
- ensemble inference on QA-filtered inputs
- optionally retrained ensemble

The outputs must include both:

- quality (`test_iou`, `test_f1`)
- filtering behavior (`filtered_pct`, `kept_pct`, rows/cell retained)

### 7. Silver truth is missing as a comparison row

Once `01_ST` and `02_ST` are present under `data/synchronized_data/BF-C2DL-HSC`, they should be evaluated as a reference row in the same comparison tables and MLflow runs.

### 8. Mixed dataset needs the same evaluation surface

The same logging structure should be used on the mixed dataset so that the best mixed-dataset ensemble result can be compared to competitors and silver truth in the same way.

---

## Rewrite Goals

### Goal A — Separate training from final comparison

Keep training experiments for debugging and model development, but add a clean final-evaluation logging surface for paper comparison.

### Goal B — Make final comparison runs uniform

Every final-evaluation run should have:

- clear run name
- clear setup identity
- explicit evaluation level
- canonical metric aliases
- threshold/filter metadata where relevant

### Goal C — Make the compare view usable

A single MLflow experiment per dataset/variant/fold should be enough to compare:

- competitors
- fusion baselines
- QA selector baselines
- full pipeline variants
- ensemble variants
- silver truth

---

## Proposed MLflow Structure

### 1. Keep training experiments

These remain useful for development:

- QA training
- ensemble training
- fusion execution diagnostics

They should still get clearer names and tags, but they are not the main paper comparison surface.

### 2. Add a dedicated comparison experiment

For each dataset / variant / fold:

```text
paper-compare-{dataset}-{variant}-fold{fold}
```

Example:

```text
paper-compare-BF-C2DL-HSC-baseline-fold2
```

This experiment should contain one run per final comparable setup.

### 3. Canonical run naming

Examples:

- `competitor__CALT-US`
- `fusion_baseline__simple`
- `fusion_only__simple`
- `full_pipeline_t0.75__simple`
- `qa_only__top1`
- `ensemble_only__unetplusplus`
- `ensemble_qa_t0.75__unetplusplus`
- `ensemble_qa_retrained_t0.75__unetplusplus`
- `silver_truth`

### 4. Required tags for comparable runs

- `pipeline_family`
- `setup_name`
- `dataset`
- `variant`
- `fold`
- `split`
- `evaluation_level`
- `qa_mode`
- `qa_threshold`
- `source_experiment`
- `source_run_id`
- `metric_schema_version`

### 5. Required canonical metrics for comparable runs

- `test_iou`
- `test_f1`
- `val_iou`
- `val_f1`
- `train_iou`
- `train_f1`
- `overall_iou`
- `overall_f1`
- `eval_count`

Optional but useful:

- `filtered_pct`
- `kept_pct`
- `rows_per_cell_mean`
- `rows_per_cell_test_mean`
- `cell_test_iou`
- `cell_test_f1`
- `image_test_iou`
- `image_test_f1`

---

## Implementation Plan

### Phase 1 — Logging normalization

1. Add shared helpers for canonical metric aliases and evaluation-level tags.
2. Update competitor evaluation logging.
3. Update fusion logging to expose canonical aliases.
4. Update ensemble final evaluation logging.
5. Update QA regression and QA filtering runs with clearer tags and run names.

### Phase 2 — Dedicated comparison experiment

1. Extend `run_ablation.py` so all final-evaluation steps can also log into `paper-compare-*`.
2. Log fusion reconstructed image-level evaluation runs there.
3. Log ensemble evaluation runs there.
4. Log competitor baselines there.
5. Log QA selector baselines there.

### Phase 3 — Missing experiments

1. Add threshold sweep orchestration for fusion and ensemble inference.
2. Add QA selector evaluation as a separate, explicit final segmentation experiment.
3. Add competitor+QA filtered comparison rows.

### Phase 4 — Dataset coverage

1. Add silver-truth ingestion from `01_ST` / `02_ST`.
2. Rebuild evaluation dataframes with silver truth as a comparison column.
3. Run the same comparison surface on the mixed dataset.

---

## Progress Snapshot

### Implemented on 2026-03-12

- Added canonical MLflow aliases (`test_iou`, `test_f1`, `val_iou`, `val_f1`, `overall_iou`, `overall_f1`) and explicit `evaluation_level` tags to the shared tracking helpers.
- Reworked ablation logging so fold-level comparison runs land in `paper-compare-{dataset}-{variant}-fold{fold}`.
- Added reconstructed-image comparison logging for fusion and QA-selector runs.
- Added explicit final-evaluation runs for ensemble variants.
- Added silver-truth ingestion support for `01_ST` / `02_ST`.

### Verified after rerunning BF-C2DL-HSC baseline fold 2

- `paper-compare-BF-C2DL-HSC-baseline-fold2` is now populated and usable for comparison.
- Fusion, QA-selector, and ensemble runs expose canonical `test_iou` / `test_f1`.
- `evaluation_level` is now visible in the compare runs.

### Still missing after the rerun

- Competitor rows still need `test_f1` alongside `test_iou`.
- Ensemble compare rows are still logged at `evaluation_level=cell_crop` because the databank parquet does not yet preserve reconstruction coordinates.
- Threshold sweeps for full-pipeline fusion and ensemble inference are not yet part of the completed run history.
- Silver-truth rows will not appear until the dataset dataframes are regenerated with the new `01_ST` / `02_ST` ingestion path.

### Next implementation slice

1. Add competitor F1 logging without changing the current label-level evaluation definition.
2. Preserve ensemble reconstruction metadata in databank generation so final compare runs can be logged at reconstructed image level.
3. Extend `run_ablation.py` to sweep inference thresholds for fusion and ensemble filtering.
4. Regenerate the BF-C2DL-HSC dataset dataframe so `SILVER-TRUTH` becomes available to the compare surface.

---

## Minimum Success Criteria

The rewrite is successful when, for fold 2 alone, the MLflow UI can answer all of these without opening code:

1. Which run is competitor vs fusion vs ensemble vs QA selector?
2. Is each number crop-level, label-level, or reconstructed image-level?
3. What are `test_iou` and `test_f1` for each setup?
4. What threshold was used?
5. How much was filtered?
6. Where is ensemble-on-filtered-inputs vs ensemble-retrained-on-filtered-inputs?
7. Where is silver truth?

---

## Immediate First Slice

The first code slice should do the following:

1. Standardize naming and tags for new runs.
2. Add canonical `test_iou` / `test_f1` aliases where final metrics already exist.
3. Add explicit MLflow logging for:
   - reconstructed fusion evaluation
   - ensemble final evaluation
4. Add an in-repo path for the comparison experiment layout.

That gets us out of the current state where the runs exist but cannot be compared cleanly.
