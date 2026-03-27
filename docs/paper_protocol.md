# Paper Protocol (Fold-Locked)

This page is the authoritative runbook for paper results.

## Scope

- Dataset split policy:
  - `fold-1`: train on `seq01`, test on `seq02`
  - `fold-2`: train on `seq02`, test on `seq01`
- Evaluation policy:
  - report full-image metrics reconstructed from crops
  - score with sparse-GT semantics (only labels present in GT are evaluated)
- Reproducibility policy:
  - deterministic data prep in DVC
  - model/fusion experiments tracked in MLflow

## Canonical Order

1. Prepare fold datasets with DVC.
2. Run Phase A through the maintained runner for each fold.
3. Run Phase B through the maintained runner for each fold.
4. Run Phase C through the maintained runner for each fold.
5. Consolidate fold outputs from `data/paper_runs/` and MLflow.

Do not use archived/mixed-split QA prediction files for paper runs.

Important:
- the maintained orchestration path is `scripts/run_ablation.py`

## Commands

### 1) Deterministic fold prep (DVC)

```bash
source .venv/bin/activate
dvc repro create_fold1@BF-C2DL-HSC create_fold2@BF-C2DL-HSC
dvc repro create_qa_crops_split_fold1@BF-C2DL-HSC create_qa_crops_split_fold2@BF-C2DL-HSC
```

Expected fold inputs:
- `data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet`
- `data/dataframes/BF-C2DL-HSC/qa_crops/fold-2_sz64_qa_dataset.parquet`

### 2) Phase A — unfiltered baselines

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/mlruns
mkdir -p data/mlflow/mlruns
```

```bash
python scripts/run_ablation.py \
  --config experiments/variants/baseline.yaml \
  --fold 1 \
  --phase A
```

```bash
python scripts/run_ablation.py \
  --config experiments/variants/baseline.yaml \
  --fold 2 \
  --phase A
```

### 3) Phase B — QA training, evaluation, and merge into paper-ready parquet

```bash
python scripts/run_ablation.py \
  --config experiments/variants/baseline.yaml \
  --fold 1 \
  --phase B
```

```bash
python scripts/run_ablation.py \
  --config experiments/variants/baseline.yaml \
  --fold 2 \
  --phase B
```

Outputs created by Phase B include:
- QA Excel predictions under `data/paper_runs/qa_results/`
- QA metrics/filtering plots under `data/paper_runs/qa_results/`
- paper-ready crop parquets under `data/dataframes/BF-C2DL-HSC/qa_crops/paper_inputs/`

### 4) Phase C — fold-locked ablation and threshold sweep

```bash
python scripts/run_ablation.py \
  --config experiments/variants/baseline.yaml \
  --fold 1 \
  --phase C
```

```bash
python scripts/run_ablation.py \
  --config experiments/variants/baseline.yaml \
  --fold 2 \
  --phase C
```

Notes:
- Phase C uses the `ablation_modes` and `qa_threshold_sweep` defined in the selected YAML config.
- The default baseline config currently sweeps `0.50,0.60,0.70,0.75`.
- If you extend the threshold list, edit the config first and then rerun Phase C.
- The runner does not pause after training; it proceeds automatically once the training process finishes. If the shell is interrupted, rerun the same command and it will resume from `.state/`.

## Mode Definitions

- `fusion_only`: Java fusion with all candidates (no QA gating).
- `full_pipeline`: QA threshold gating, then Java fusion on survivors.
- `qa_only`: choose QA-best candidate per cell (after threshold fallback), no fusion voting.

`qa_only` is a control to isolate QA ranking without fusion.
It is not the same as "re-score each competitor after QA filtering".

## Planned Extension (Requested)

Additional experiment to add:
- For each competitor separately, apply QA threshold filtering to that competitor's cells only, reconstruct full images, and compare against unfiltered competitor baseline.

Why this is useful:
- isolates whether QA filtering alone improves a single model, independent of fusion voting.
