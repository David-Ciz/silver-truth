# Ablation Experiment Plan
_Updated: 2026-03-12 | Dataset: BF-C2DL-HSC (primary) | Folds: fold-1 and fold-2_

Cross-references:
- [`docs/paper_protocol.md`](paper_protocol.md) — step-by-step command runbook
- [`docs/ablation_results_and_insights.md`](ablation_results_and_insights.md) — result summary and current interpretation
- [`docs/ablation_split_sizes.md`](ablation_split_sizes.md) — actual labeled image/cell counts per split
- [`PAPER_STATUS.md`](../PAPER_STATUS.md) — phase tracking and completion gates

---

## Design Philosophy

> "The ablation should just make the pipeline easier to run, but if I wanted I could use the CLI and run the individual commands manually."

Each experiment is a **sequence of existing CLI commands**. Nothing new is invented unless a genuine gap exists in the current CLI. Every step is independently runnable and inspectable.

The maintained orchestrator for these steps is `scripts/run_ablation.py`, but the manual commands below are still useful for understanding and debugging each phase.

### Natural experiment order

When a new dataset arrives it comes with competitors. The experiments follow the natural progression of what requires training:

1. **No training needed** — evaluate competitors, run Java fusions, train + evaluate the ensemble (learned fusion)
2. **QA training** — train QA model, evaluate its validity
3. **QA-filtered re-runs** — re-run competitors, all fusions (Java + ensemble) on QA-filtered inputs

This is the order the experiments are written below.

### The ensemble is a fusion

The deep-learning ensemble (`silver-ensemble`) is another fusion strategy — it learns to combine competitor crops rather than using a fixed voting rule. It is treated exactly like the Java fusion strategies throughout:
- run unfiltered as a baseline (Phase A)
- run on QA-filtered input as the full-pipeline variant (Phase B)

---

## Evaluation Principles

### Sparse GT — evaluate label by label
GT annotations are sparse: not every cell in an image has a mask. The evaluator (`metrics/evaluation_logic.py:run_evaluation`) iterates only labels present in the GT image and computes per-label IoU. The existing `silver-evaluation evaluate-competitor` command already does this correctly.

### Full-image IoU — not crop IoU
Crops are 64×64 patches. A crop that clips a cell at the edge gives a smaller GT region → wrong IoU. **All paper numbers use full-image IoU.** For competitor evaluation, use the whole-image parquets directly. For anything involving crops (Java fusion, ensemble), the crop must be placed back into the full image before scoring.

### Crop → full-image reconstruction
Place a fused/predicted crop back into the full image at `(crop_y_start:crop_y_end, crop_x_start:crop_x_end)`, then run the standard per-label evaluator. Cells within the same image have non-overlapping GT labels, so pixel-level crop overlap does not affect evaluation — each cell is scored independently.

- **Java fusion**: `reconstruction.reconstruct_full_images_from_paths` in `ensemble/reconstruction.py` — already implemented, takes a fused-path column + crop coordinate columns.
- **Ensemble**: `reconstruction.reconstruct_full_images_from_arrays` — already called inside `ensemble.generate_evaluation` when reconstruction metadata is present.

### Fold-lock
- fold-1: train seq01, test seq02
- fold-2: train seq02, test seq01
- QA models must be trained on the fold-matched parquet (see `docs/paper_protocol.md`).

---

## What Already Exists

### DVC data prep stages
| Stage | Output |
|---|---|
| `create_fold1@BF-C2DL-HSC` | `data/dataframes/BF-C2DL-HSC/whole_image/BF-C2DL-HSC_split_fold-1.parquet` |
| `create_fold2@BF-C2DL-HSC` | `data/dataframes/BF-C2DL-HSC/whole_image/BF-C2DL-HSC_split_fold-2.parquet` |
| `create_qa_crops_split_fold1@BF-C2DL-HSC` | `data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet` |
| `create_qa_crops_split_fold2@BF-C2DL-HSC` | `data/dataframes/BF-C2DL-HSC/qa_crops/fold-2_sz64_qa_dataset.parquet` |
| `generate_job_files_fold1` | `data/job_files/BF-C2DL-HSC/fold-1/…_job_file.txt` (whole-image paths) |
| `generate_job_files_fold2` | `data/job_files/BF-C2DL-HSC/fold-2/…_job_file.txt` |

Run all prep:
```bash
source .venv/bin/activate
dvc repro create_fold1@BF-C2DL-HSC create_fold2@BF-C2DL-HSC \
          create_qa_crops_split_fold1@BF-C2DL-HSC create_qa_crops_split_fold2@BF-C2DL-HSC \
          generate_job_files_fold1 generate_job_files_fold2
```

### CLI commands
| Command | Group | Purpose |
|---|---|---|
| `silver-evaluation evaluate-competitor` | `silver-evaluation` | Full-image label-by-label IoU/F1 on whole-image parquets |
| `silver-fusion run-fusion-crops` | `silver-fusion` | Run Java fusion on crop parquet; evaluates at crop level |
| `silver-ensemble build-databank` | `silver-ensemble` | Build ensemble databank from QA crop parquet (optionally with QA gating) |
| `silver-ensemble ensemble-experiment` | `silver-ensemble` | Train ensemble model (U-Net/UnetPlusPlus) with MLflow logging |
| `silver-ensemble evaluate-checkpoint` | `silver-ensemble` | Evaluate a checkpoint; calls `reconstruct_full_images_from_arrays` when reconstruction metadata present |
| `silver-ensemble evaluate-best-checkpoint` | `silver-ensemble` | Pick newest checkpoint in a dir and evaluate |
| `silver-qa cnn train` | `silver-qa` | Train QA ResNet50; outputs `.pt` model + Excel predictions |
| `silver-evaluation evaluate-qa-model` | `silver-evaluation` | QA regression metrics (MAE, RMSE, R², Pearson r, Spearman rho) |
| `silver-evaluation evaluate-qa-filtering` | `silver-evaluation` | QA as binary filter: precision/recall/F1 per threshold |
| `silver-evaluation merge-qa-predictions` | `silver-evaluation` | Add `predicted_jaccard_index` column to a parquet from QA Excel |

### Key source modules
- `metrics/evaluation_logic.py:run_evaluation` — authoritative full-image evaluator
- `metrics/metrics.py:calculate_jaccard_scores` — per-label IoU, inner loop of `run_evaluation`
- `ensemble/reconstruction.py:reconstruct_full_images_from_paths` — place on-disk fused crops back into full image and score; **already handles the Java fusion evaluation gap**
- `ensemble/reconstruction.py:reconstruct_full_images_from_arrays` — same for in-memory arrays; called by `ensemble.generate_evaluation`
- `fusion/crops_experiment.py:run_crops_fusion_experiment` — Java fusion on crops; writes per-cell fused TIFs + output parquet with fused paths and crop coordinates
- `qa/filtering_evaluation.py:run_qa_filtering_evaluation` — QA filter confusion metrics

---

## Implemented Since Draft

The two gaps from the initial draft are now closed and were used in the executed baseline ablation run:

| Implemented piece | Where it lives | Notes |
|---|---|---|
| Full-image evaluation of reconstructed fusion outputs | `silver-evaluation evaluate-fusion-crops` in `src/silver_truth/cli/evaluation.py` | Wraps `reconstruct_full_images_from_paths` and writes per-image IoU/F1 CSVs. |
| Parquet filtering for `qa_only` / `full_pipeline` | `silver-evaluation filter-parquet` in `src/silver_truth/cli/evaluation.py` | Used by `scripts/run_ablation.py` for Phase C preprocessing. |

Baseline variant execution status:
- fold-1: complete
- fold-2: complete
- state files: `.state/BF-C2DL-HSC_baseline_fold-1.json`, `.state/BF-C2DL-HSC_baseline_fold-2.json`

---

## Phase A — Unfiltered Baselines (no QA training required)

### Experiment 1 — Competitor Baselines

**Goal**: Full-image IoU/F1 for each of the 5 competitors on both folds.

**Status**: ✅ Fully implemented. First thing to run on a new dataset after DVC prep.

```bash
source .venv/bin/activate
mkdir -p data/paper_runs/baselines

silver-evaluation evaluate-competitor \
  data/dataframes/BF-C2DL-HSC/whole_image/BF-C2DL-HSC_split_fold-1.parquet \
  --output data/paper_runs/baselines/BF-C2DL-HSC_fold-1_competitors.csv

silver-evaluation evaluate-competitor \
  data/dataframes/BF-C2DL-HSC/whole_image/BF-C2DL-HSC_split_fold-2.parquet \
  --output data/paper_runs/baselines/BF-C2DL-HSC_fold-2_competitors.csv
```

**Outputs**: Per-label and per-image IoU/F1 for CALT-US, DREX-US, KIT-Sch-GE, KTH-SE, MU-Lux-CZ on each fold.

**Paper content**: Competitor baseline table (test-fold IoU, both folds + average).

---

### Experiment 2 — Java Fusion Baselines

**Goal**: Full-image IoU/F1 for each Java fusion strategy (SIMPLE, MAJORITY_FLAT, THRESHOLD_FLAT, BIC_FLAT_VOTING) on both folds. No training required — just call the JAR.

**Status**: ✅ Implemented and executed for both folds. Full-image evaluation is now handled by `silver-evaluation evaluate-fusion-crops`.

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/mlruns

silver-fusion run-fusion-crops \
  --qa-parquet data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet \
  --models SIMPLE --models MAJORITY_FLAT --models THRESHOLD_FLAT --models BIC_FLAT_VOTING \
  --output-dir data/paper_runs/fusion/fold-1 \
  --mlflow-experiment fusion-baseline-BF-C2DL-HSC-fold1

silver-fusion run-fusion-crops \
  --qa-parquet data/dataframes/BF-C2DL-HSC/qa_crops/fold-2_sz64_qa_dataset.parquet \
  --models SIMPLE --models MAJORITY_FLAT --models THRESHOLD_FLAT --models BIC_FLAT_VOTING \
  --output-dir data/paper_runs/fusion/fold-2 \
  --mlflow-experiment fusion-baseline-BF-C2DL-HSC-fold2

# Full-image evaluation example:
# silver-evaluation evaluate-fusion-crops \
#   data/paper_runs/fusion/fold-1/<model>/<model>_with_fused.parquet \
#   --output data/paper_runs/fusion/fold-1/<model>_fullimage_eval.csv
```

**Paper content**: Java fusion baseline table (test-fold IoU, both folds + average).

---

### Experiment 3 — Ensemble (Learned Fusion) Baseline

**Goal**: Full-image IoU/F1 for the deep-learning ensemble trained and evaluated unfiltered on both folds. The ensemble is trained on the same crop parquet as the Java fusions — it learns to combine competitor crops rather than using a fixed voting rule.

**Status**: ✅ Executed for both folds. Baseline ensemble checkpoints and test-set evaluation parquets are present under `data/paper_runs/ensemble/baseline/fold-{1,2}/`.

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/mlruns
mkdir -p data/paper_runs/ensemble

# Build ensemble databank (unfiltered) — fold 1
silver-ensemble build-databank \
  --dataset-name BF-C2DL-HSC \
  --qa-parquet-path data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet \
  --version C1

# Build ensemble databank (unfiltered) — fold 2
silver-ensemble build-databank \
  --dataset-name BF-C2DL-HSC \
  --qa-parquet-path data/dataframes/BF-C2DL-HSC/qa_crops/fold-2_sz64_qa_dataset.parquet \
  --version C1

# Train ensemble — fold 1
silver-ensemble ensemble-experiment \
  --name ensemble-baseline-BF-C2DL-HSC-fold1 \
  --parquet-file <databank_fold1.parquet> \
  --model-type UnetPlusPlus \
  --max-epochs 100

# Train ensemble — fold 2
silver-ensemble ensemble-experiment \
  --name ensemble-baseline-BF-C2DL-HSC-fold2 \
  --parquet-file <databank_fold2.parquet> \
  --model-type UnetPlusPlus \
  --max-epochs 100

# Evaluate best checkpoint — full-image reconstruction happens automatically
# inside evaluate-checkpoint when recon metadata columns are present
silver-ensemble evaluate-best-checkpoint \
  --checkpoints-dir <mlflow_artifacts_or_checkpoint_dir_fold1> \
  --databank-path <databank_fold1.parquet> \
  --split-type test

silver-ensemble evaluate-best-checkpoint \
  --checkpoints-dir <mlflow_artifacts_or_checkpoint_dir_fold2> \
  --databank-path <databank_fold2.parquet> \
  --split-type test
```

**Paper content**: Ensemble baseline row alongside Java fusion strategies in the fusion baseline table.

---

## Phase B — QA Training + Validity

### Experiment 4 — QA Model Training + Validity

**Goal**: Prove the QA model learns a meaningful quality signal. Required before any QA-filtered experiments.

**Status**: ✅ Executed for both folds. Outputs are in `data/paper_runs/qa_models/` and `data/paper_runs/qa_results/`.

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/mlruns
mkdir -p data/paper_runs/qa_models data/paper_runs/qa_results

# Train QA — fold 1
silver-qa cnn train \
  --parquet-file data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet \
  --output-model data/paper_runs/qa_models/qa_resnet50_fold-1.pt \
  --output-excel data/paper_runs/qa_results/qa_predictions_fold-1.xlsx \
  --model-type resnet50 --input-channels 0,1 \
  --mlflow-experiment qa-cnn-BF-C2DL-HSC-fold1

# Train QA — fold 2
silver-qa cnn train \
  --parquet-file data/dataframes/BF-C2DL-HSC/qa_crops/fold-2_sz64_qa_dataset.parquet \
  --output-model data/paper_runs/qa_models/qa_resnet50_fold-2.pt \
  --output-excel data/paper_runs/qa_results/qa_predictions_fold-2.xlsx \
  --model-type resnet50 --input-channels 0,1 \
  --mlflow-experiment qa-cnn-BF-C2DL-HSC-fold2

# Regression quality (MAE, RMSE, R², Pearson r, Spearman rho)
silver-evaluation evaluate-qa-model \
  data/paper_runs/qa_results/qa_predictions_fold-1.xlsx \
  --output-dir data/paper_runs/qa_results/fold-1_metrics \
  --mlflow-experiment qa-eval-BF-C2DL-HSC-fold1

silver-evaluation evaluate-qa-model \
  data/paper_runs/qa_results/qa_predictions_fold-2.xlsx \
  --output-dir data/paper_runs/qa_results/fold-2_metrics \
  --mlflow-experiment qa-eval-BF-C2DL-HSC-fold2

# Filtering validity (precision/recall/F1 as a binary filter at each threshold)
silver-evaluation evaluate-qa-filtering \
  data/paper_runs/qa_results/qa_predictions_fold-1.xlsx \
  --thresholds 0.50,0.60,0.70,0.75,0.80,0.85,0.90 \
  --output-dir data/paper_runs/qa_results/fold-1_filtering \
  --mlflow-experiment qa-filtering-BF-C2DL-HSC-fold1

silver-evaluation evaluate-qa-filtering \
  data/paper_runs/qa_results/qa_predictions_fold-2.xlsx \
  --thresholds 0.50,0.60,0.70,0.75,0.80,0.85,0.90 \
  --output-dir data/paper_runs/qa_results/fold-2_filtering \
  --mlflow-experiment qa-filtering-BF-C2DL-HSC-fold2
```

**Outputs**:
- `data/paper_runs/qa_models/qa_resnet50_fold-{1,2}.pt`
- `data/paper_runs/qa_results/qa_predictions_fold-{1,2}.xlsx`
- `data/paper_runs/qa_results/fold-{1,2}_metrics/` — regression metrics + scatter plots
- `data/paper_runs/qa_results/fold-{1,2}_filtering/` — confusion-matrix filtering metrics

**Paper content**: QA regression table (MAE/RMSE/R²/Pearson r per fold). QA filtering validity table/plot vs threshold.

### Oracle QA Sanity Check

Before judging the learned QA model, run the same gating logic with the **true per-cell IoU against GT** as an oracle score.

Setup:
- score each competitor crop with its measured GT IoU instead of `predicted_jaccard_index`
- sweep thresholds on that oracle score
- re-run the same downstream selection/fusion pipeline on the surviving candidates
- compare the result to the unfiltered baseline

Why this matters:
- it tests the core hypothesis directly: if we could perfectly detect bad segmentations, would filtering actually improve fusion?
- it gives an upper bound on what QA-based filtering can achieve on this dataset
- it separates "the QA model is not accurate enough" from "filtering is not the right lever"

Interpretation:
- if no oracle threshold beats the unfiltered baseline, then the filtering premise is weak
- if an oracle threshold helps, then the real objective is to learn a QA model that approximates that oracle

For any paper-facing claim, threshold selection still has to be fold-locked: choose it on train/validation GT only, then report it on the held-out test fold.

---

## Phase C — QA-Filtered Re-runs (Core Ablation)

All experiments in this phase use the QA predictions from Phase B. First, merge predictions into the crop parquets:

```bash
source .venv/bin/activate
mkdir -p data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs

cp data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet \
   data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_paper_base.parquet
cp data/dataframes/BF-C2DL-HSC/qa_crops/fold-2_sz64_qa_dataset.parquet \
   data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-2_paper_base.parquet

silver-evaluation merge-qa-predictions \
  data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_paper_base.parquet \
  data/paper_runs/qa_results/qa_predictions_fold-1.xlsx \
  --output data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_paper_ready.parquet

silver-evaluation merge-qa-predictions \
  data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-2_paper_base.parquet \
  data/paper_runs/qa_results/qa_predictions_fold-2.xlsx \
  --output data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-2_paper_ready.parquet
```

**Parquet filtering helpers** are already implemented as `silver-evaluation filter-parquet`. The pandas sketch below is only illustrative:

```python
import pandas as pd

def filter_parquet_qa_only(path, out):
    """Keep only top-1 predicted_jaccard_index row per cell."""
    df = pd.read_parquet(path)
    top1 = (df.sort_values("predicted_jaccard_index", ascending=False)
              .groupby(["campaign_number", "original_image_key", "label"], as_index=False)
              .first())
    top1.to_parquet(out)

def filter_parquet_full_pipeline(path, out, threshold=0.75):
    """Keep rows >= threshold; fall back to top-1 for cells with no passing row."""
    df = pd.read_parquet(path)
    passing = df[df["predicted_jaccard_index"] >= threshold]
    key_cols = ["campaign_number", "original_image_key", "label"]
    covered = set(map(tuple, passing[key_cols].values))
    fallback = (df[~df[key_cols].apply(tuple, axis=1).isin(covered)]
                  .sort_values("predicted_jaccard_index", ascending=False)
                  .groupby(key_cols, as_index=False).first())
    pd.concat([passing, fallback]).to_parquet(out)
```

---

### Experiment 5 — Core Ablation (fusion_only / qa_only / full_pipeline)

**Goal**: Isolate the contribution of QA filtering vs. fusion voting.

**Status**: ✅ Executed for the baseline variant at `t=0.75` on both folds.

**Mode definitions**:

| Mode | Input parquet | Fusion? |
|---|---|---|
| `fusion_only` | `fold-{n}_paper_ready.parquet` as-is | Yes — Java fusion |
| `qa_only` | `filter_parquet_qa_only(...)` — top-1 per cell, no fusion | No — evaluate selected crop directly |
| `full_pipeline` | `filter_parquet_full_pipeline(..., threshold=0.75)` | Yes — Java fusion on surviving rows |
| `ensemble_only` | `fold-{n}_paper_ready.parquet` as-is | Yes — ensemble (trained in Exp 3) |
| `ensemble_qa` (primary) | filtered databank built from `full_pipeline` parquet, evaluated with **unfiltered checkpoint** from Exp 3 | Yes — same trained model, different inputs |
| `ensemble_qa_retrained` (secondary) | same filtered databank, but **retrained from scratch** on it | Yes — joint QA+ensemble optimization |

**`ensemble_qa` primary** uses the checkpoint already trained in Exp 3 — the ensemble and QA were trained independently, yet we check whether QA-filtered inputs improve the ensemble's output. This cleanly isolates whether the two components help each other without any joint training signal.

**`ensemble_qa_retrained` secondary** retrains the ensemble on QA-filtered inputs from scratch. This tests the upper bound of joint QA+ensemble optimization and is a valid separate row in the ablation table.

```bash
# fusion_only
silver-fusion run-fusion-crops \
  --qa-parquet data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_paper_ready.parquet \
  --models BIC_FLAT_VOTING \
  --output-dir data/paper_runs/ablation/fold-1/fusion_only \
  --mlflow-experiment ablation-fusion-only-fold1

# qa_only — evaluate top-1 selected crops directly (full-image reconstruction)
# filter first, then: silver-evaluation evaluate-fusion-crops <filtered_parquet>

# full_pipeline — filter then fuse
silver-fusion run-fusion-crops \
  --qa-parquet data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_full_pipeline_t0.75.parquet \
  --models BIC_FLAT_VOTING \
  --output-dir data/paper_runs/ablation/fold-1/full_pipeline_t0.75 \
  --mlflow-experiment ablation-full-pipeline-t0.75-fold1

# ensemble_qa (primary) — build filtered databank, evaluate with EXISTING unfiltered checkpoint
silver-ensemble build-databank \
  --dataset-name BF-C2DL-HSC \
  --qa-parquet-path data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_full_pipeline_t0.75.parquet \
  --version C1
silver-ensemble evaluate-best-checkpoint \
  --checkpoints-dir <checkpoint_dir_fold1_from_exp3> \
  --databank-path <filtered_databank_fold1.parquet> \
  --split-type test

# ensemble_qa_retrained (secondary) — retrain ensemble from scratch on the filtered databank
silver-ensemble ensemble-experiment \
  --name ensemble-qa-retrained-BF-C2DL-HSC-fold1 \
  --parquet-file <filtered_databank_fold1.parquet> \
  --model-type UnetPlusPlus \
  --max-epochs 100
silver-ensemble evaluate-best-checkpoint \
  --checkpoints-dir <checkpoint_dir_retrained_fold1> \
  --databank-path <filtered_databank_fold1.parquet> \
  --split-type test
```

**Paper content**: Main ablation table — all modes side by side (test-fold IoU/F1, both folds + average).

---

### Experiment 6 — Threshold Sweep

**Goal**: Show how `full_pipeline` (Java fusion) and `ensemble_qa` IoU + filtering rate vary with the QA threshold. Justify t=0.75.

**Status**: ⏭️ Not run yet. This is now the highest-priority follow-up because the completed `t=0.75` run does not justify that threshold.

For each t ∈ {0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90}:
1. `filter_parquet_full_pipeline(fold-{n}_paper_ready.parquet, ..., threshold=t)`
2. `silver-fusion run-fusion-crops` on the filtered parquet
3. `silver-evaluation evaluate-fusion-crops` for full-image scores

Filtering rate per threshold is already available from the `evaluate-qa-filtering` output in Experiment 4 — cross-reference those tables.

**Paper content**: Dual-axis plot — IoU vs threshold + % cells filtered vs threshold.

---

### Experiment 7 — Competitor + QA Filter

**Goal**: Show QA filtering improves a single competitor in isolation (no fusion). Proves QA is useful independent of the fusion component.

**Status**: ⏭️ Not run yet. No new code is needed — `evaluate-competitor` already handles a reduced parquet, so absent labels simply don't contribute a score.

```python
# Filter script
import pandas as pd

df = pd.read_parquet("data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/fold-1_paper_ready.parquet")
threshold = 0.75
out_dir = "data/paper_runs/competitor_qa_filter/fold-1"

for competitor in df["competitor"].unique():
    comp_df = df[df["competitor"] == competitor].copy()
    filtered_df = comp_df[comp_df["predicted_jaccard_index"] >= threshold]
    comp_df.to_parquet(f"{out_dir}/{competitor}_unfiltered.parquet")
    filtered_df.to_parquet(f"{out_dir}/{competitor}_qa_filtered.parquet")
```

```bash
silver-evaluation evaluate-competitor \
  data/paper_runs/competitor_qa_filter/fold-1/CALT-US_unfiltered.parquet \
  --output data/paper_runs/competitor_qa_filter/fold-1/CALT-US_unfiltered_eval.csv

silver-evaluation evaluate-competitor \
  data/paper_runs/competitor_qa_filter/fold-1/CALT-US_qa_filtered.parquet \
  --output data/paper_runs/competitor_qa_filter/fold-1/CALT-US_filtered_eval.csv
# ... repeat for all competitors and fold-2
```

**Paper content**: Table — unfiltered vs QA-filtered IoU per competitor per fold.

---

## Implemented CLI Wiring

This gap is closed.

`silver-evaluation evaluate-fusion-crops` and `silver-evaluation filter-parquet` now live in `src/silver_truth/cli/evaluation.py`, and `reconstruct_full_images_from_paths` already handles both `recon_crop_*` and plain `crop_*` coordinate naming. No extra CLI work is needed for the baseline BF-C2DL-HSC ablation path.

---

## Executed Baseline Run — 2026-03-11

Baseline ablation was executed through `scripts/run_ablation.py` for `experiments/variants/baseline.yaml` on both folds. The result roots are:

- `data/paper_runs/baselines/`
- `data/paper_runs/qa_results/`
- `data/paper_runs/ensemble/baseline/`
- `data/paper_runs/ablation/baseline/`

### Main test-set snapshot

These are the most decision-relevant test-fold IoU numbers from the completed run. Fusion and competitor rows below use the label-level test IoU already written by the existing evaluators and summary files.

| Row | fold-1 | fold-2 | average |
|---|---:|---:|---:|
| Best competitor (`CALT-US`) | 0.8526 | 0.8953 | 0.8739 |
| `fusion_only / SIMPLE` | 0.8444 | 0.8875 | 0.8659 |
| `fusion_only / MAJORITY_FLAT` | 0.8419 | 0.8733 | 0.8576 |
| `ensemble_only` | 0.8469 | 0.8433 | 0.8451 |
| `full_pipeline / SIMPLE` (`t=0.75`) | 0.8280 | 0.8007 | 0.8144 |
| `ensemble_qa` (`t=0.75`) | 0.8354 | 0.7931 | 0.8142 |
| `ensemble_qa_retrained` (`t=0.75`) | 0.8401 | 0.7985 | 0.8193 |
| `full_pipeline / THRESHOLD_FLAT` | 0.7986 | 0.8001 | 0.7993 |
| `full_pipeline / MAJORITY_FLAT` | 0.4212 | 0.0070 | 0.2141 |

`qa_only` is easiest to interpret from the reconstructed full-image output:

| Row | fold-1 image-level IoU | fold-2 image-level IoU | average |
|---|---:|---:|---:|
| `qa_only / top-1` | 0.8522 | 0.7950 | 0.8236 |

### What the run says

- `CALT-US` remains the strongest single competitor on both folds. None of the ablation branches beat it on both test folds.
- `SIMPLE` is still the strongest fusion strategy. It stays close to `CALT-US` unfiltered, while every QA-filtered branch lands clearly lower.
- Fold-1 tests on seq02 and is the harder side of the BF-C2DL-HSC split. Most methods score lower on fold-1 than on fold-2, which is a reminder not to over-read single-fold wins.
- QA generalization is the weak point. Test regression is poor on both folds, and especially weak on fold-2:

| Fold | test R² | test MAE | test RMSE | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|---:|
| fold-1 | -0.2251 | 0.0727 | 0.0886 | 0.4088 | 0.2891 |
| fold-2 | -1.1536 | 0.1587 | 0.1776 | 0.1345 | 0.1022 |

- The current `t=0.75` gate is not supported by the QA validation outputs. In the saved filtering reports, the best-F1 threshold is `0.50` on both folds, not `0.75`.
- `t=0.75` removes too much fusion context on the test split. In `full_pipeline`, the mean retained rows per cell drop from `4.88` to `2.57` on fold-1 test and all the way to `1.03` on fold-2 test. On fold-2 test, `249/255` cells keep only one crop after thresholding.
- That retention pattern explains the failure mode of `MAJORITY_FLAT`: it loses the multi-competitor vote it depends on. The collapse is not subtle: `0.8419 → 0.4212` on fold-1 and `0.8733 → 0.0070` on fold-2.
- `THRESHOLD_FLAT` and `BIC_FLAT_VOTING` are numerically identical in both folds and both branches. They actually improve after QA filtering (`+0.0674` on fold-1, `+0.0242` on fold-2), but they still trail `SIMPLE` and the best competitor.
- The ensemble does not benefit from the current QA stage. `ensemble_only` averages `0.8451`, `ensemble_qa` drops to `0.8142`, and retraining on the filtered databank only recovers to `0.8193`.
- `qa_only` is informative even though it is not the main paper metric. On fold-2 it is almost tied with filtered `SIMPLE`, which suggests the current QA branch is acting more like single-prediction selection than a useful pre-fusion gate.

### Recommended next steps

1. Run the threshold sweep from Experiment 6 before making any paper claim about QA helping the pipeline. Start with `0.50, 0.60, 0.70, 0.75`, not just `0.75+`.
2. Treat `fusion_only / SIMPLE` as the unfiltered fusion reference row for the paper. It is the strongest fusion baseline in the completed run.
3. Run Experiment 7 next, starting with `CALT-US` and `MU-Lux-CZ`. If QA helps anywhere right now, it is more likely to show up as competitor selection than as pre-fusion pruning.
4. If QA is meant to remain a core claim, improve or recalibrate the QA model before re-running the full pipeline. Fold-2 test regression is too weak for a hard threshold to be trusted.
5. Keep `ensemble_qa_retrained` as a secondary result unless a lower threshold changes the picture. In the current run it does not beat the unfiltered ensemble.
6. Add bootstrap confidence intervals and paired Wilcoxon tests only after the final threshold and final ablation rows are fixed.

---

## Minimal Result Package (Paper Claims Gate)

**Phase A — unfiltered baselines:**
- [x] Competitor baseline table (test-fold IoU/F1 for all 5, both folds + average) — Exp 1
- [x] Java fusion baseline table (test-fold IoU/F1 for 4 strategies, both folds + average) — Exp 2
- [x] Ensemble baseline row (test-fold IoU/F1, both folds + average) — Exp 3

**Phase B — QA validity:**
- [x] QA regression metrics (MAE, RMSE, R², Pearson r) for both folds — Exp 4
- [x] QA filtering validity at t=0.75 (precision/recall/F1/FPR/FNR, % filtered) for both folds — Exp 4

**Phase C — QA-filtered:**
- [x] Ablation table (fusion_only / qa_only / full_pipeline / ensemble_only / ensemble_qa / ensemble_qa_retrained at t=0.75, both folds + average) — Exp 5
- [ ] Threshold sweep table (IoU/F1 + % filtered for t ∈ 7 values, both folds + average) — Exp 6
- [ ] Competitor+QA-filter table (unfiltered vs filtered IoU per competitor, both folds) — Exp 7

**Statistics:**
- [ ] Bootstrap 95% CI on test-set IoU for all ablation rows
- [ ] Paired Wilcoxon p-value: full_pipeline vs fusion_only
- [ ] Paired Wilcoxon p-value: full_pipeline vs best competitor

**Reproducibility:**
- [ ] Git SHA, input parquet paths, QA model paths, commands used for each result
