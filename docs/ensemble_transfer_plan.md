# Ensemble Transfer Plan
_Updated: 2026-03-13 | Scope: BF-C2DL-MuSC -> BF-C2DL-HSC | Paper-safe fold protocol_

This plan replaces the earlier mixed-first transfer idea.

The goal is not to assume `BF-C2DL-MuSC` is a good source domain. The first job is to verify that the current ensemble setup is viable on MuSC itself, and to check whether `64x64` crops are large enough before spending time on transfer.

---

## Working Rules

- Do not use mixed-trained model results for paper claims.
- Fold-locked evaluation remains the reporting path:
  - `fold-1`: train on seq01, test on seq02
  - `fold-2`: train on seq02, test on seq01
- Mixed MuSC artifacts may still appear as a technical prerequisite in DVC because `create_qa_crops_base` is built from the mixed whole-image parquet. That is an implementation detail, not a paper evaluation regime.

---

## Main Question Order

Do these in order:

1. Can the current ensemble setup learn anything useful on MuSC itself?
2. Are `64x64` crops adequate on MuSC?
3. If MuSC looks learnable, does MuSC pretraining improve HSC fold performance?

If the answer to (1) is "no", do not run transfer yet.

---

## Phase 0: Prepare MuSC Data

### DVC steps

These produce the MuSC whole-image and QA crop parquets needed for fold-safe experiments:

```bash
dvc repro create_mixed@BF-C2DL-MuSC
dvc repro create_fold1@BF-C2DL-MuSC create_fold2@BF-C2DL-MuSC
dvc repro create_qa_crops_base@BF-C2DL-MuSC
dvc repro create_qa_crops_split_fold1@BF-C2DL-MuSC create_qa_crops_split_fold2@BF-C2DL-MuSC
```

Why `create_mixed@BF-C2DL-MuSC` is still needed:
- `create_qa_crops_base@BF-C2DL-MuSC` currently depends on the mixed whole-image dataframe
- this is only a crop-generation dependency, not a training/evaluation recommendation

Expected outputs:
- `data/dataframes/BF-C2DL-MuSC/whole_image/BF-C2DL-MuSC_split_fold-1.parquet`
- `data/dataframes/BF-C2DL-MuSC/whole_image/BF-C2DL-MuSC_split_fold-2.parquet`
- `data/dataframes/BF-C2DL-MuSC/qa_crops/fold-1_sz64_qa_dataset.parquet`
- `data/dataframes/BF-C2DL-MuSC/qa_crops/fold-2_sz64_qa_dataset.parquet`

---

## Phase 1: Check Whether 64x64 Crops Are Viable On MuSC

This must happen before transfer.

Questions:
- are cells frequently clipped at crop borders?
- does the centered object usually fit comfortably inside `64x64`?
- do MuSC cells look systematically larger than HSC cells?

Minimum manual check:
- inspect a random sample of MuSC crops from `data/qa_crops/BF-C2DL-MuSC/sz64/`
- inspect failure-looking examples from both folds

Recommended quantitative check before manual review:

```bash
silver-preprocessing segmentation-size-stats \
  data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG \
  data/synchronized_data/BF-C2DL-MuSC/02_GT/SEG \
  --crop-size 64 --crop-size 96 --crop-size 128
```

Interpretation:
- use the GT bbox fit rate as a fast rejection test
- if many MuSC GT objects have bbox height or width above `64`, `64x64` is not a safe default
- still inspect crops manually, but do not ignore a bad quantitative fit rate

### Findings Recorded On 2026-03-13

Synchronization:
- MuSC synchronized segmentations were verified against TRA for `01_GT/SEG`, `02_GT/SEG`, and all MuSC competitor `01_RES` / `02_RES` folders.
- No desynchronized MuSC folders were found in that verification pass.

Quantitative size findings from GT masks:
- MuSC overall: `515` GT cells across `100` GT masks.
- MuSC mean area: `1227.97 px`
- MuSC median area: `941 px`
- MuSC max area: `4775 px`
- MuSC mean bbox: `68.4 x 62.9`
- MuSC max bbox: `375 x 411`
- MuSC bbox fit rate at `64`: `53.6%` (`239 / 515` cells exceed `64` in height or width)
- MuSC bbox fit rate at `96`: `69.3%`
- MuSC bbox fit rate at `128`: `78.6%`

Per-sequence note:
- MuSC `01`: `70.0%` fit at `64`
- MuSC `02`: `41.6%` fit at `64`
- seq02 is the stronger warning signal; `64` is particularly weak there

HSC comparison reference:
- HSC overall mean area: `317.21 px`
- HSC max bbox dimension: `38`
- HSC bbox fit rate at `64`: `100%`

Interpretation:
- MuSC cells are substantially larger than HSC cells.
- MuSC mean cell area is about `3.9x` HSC mean area.
- `64x64` is acceptable for HSC but not for MuSC.
- This is not a borderline result. `64` is not a safe default for MuSC transfer experiments.

Decision rule:
- if clipping looks rare, keep `64`
- if many target cells touch crop boundaries, stop and run a crop-size branch before any transfer work

Current decision:
- do not proceed with MuSC transfer on `64x64`
- open a crop-size branch first

Recommended crop-size follow-up only if needed:
- compare `64` vs `96` or `128`
- do this on MuSC first, not HSC

Important:
- do not fork crop size casually
- only branch if the current crop size is visibly invalid

Recommended next steps after the 2026-03-13 findings:
1. Do not run `ensemble_ref_musc` on `sz64` as the main MuSC baseline path.
2. Create a MuSC crop-size branch and evaluate at least `96` and `128`.
3. Prefer `128` as the first candidate, because `96` still leaves `30.7%` of MuSC GT boxes oversized.
4. Regenerate MuSC QA crops/parquets at the chosen larger crop size.
5. Re-run the same size check on the regenerated setting and spot-check real crops for border clipping.
6. Only after a larger crop size looks valid, run MuSC fold baselines.
7. Only after MuSC fold baselines look healthy, revisit MuSC -> HSC transfer.

---

## Phase 2: Establish MuSC Baselines First

Before calling MuSC a useful pretraining source, train the ensemble on MuSC itself.

### Runner

Use the existing config-driven runner:
- config: `experiments/variants/ensemble_ref_musc.yaml`

### Commands

Fold 1:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_ablation.py \
  --config experiments/variants/ensemble_ref_musc.yaml \
  --fold 1
```

Fold 2:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_ablation.py \
  --config experiments/variants/ensemble_ref_musc.yaml \
  --fold 2
```

Runner behavior:
- the current runner continues automatically after training finishes
- only rerun the command if the shell was interrupted and you want to resume from `.state/`

### What to record

For each MuSC fold:
- test IoU
- test F1
- test cell count / reconstructed image count
- qualitative reconstruction sanity

Decision rule:
- if MuSC fold performance is weak or unstable, do not trust it as a transfer source yet
- if MuSC looks healthy, proceed to transfer

---

## Phase 3: Only Then Run Fold-Locked Transfer

After MuSC baselines exist, use them as initialization for HSC on the matching fold.

### Runner

Use:
- pretrain config: `experiments/variants/ensemble_ref_musc.yaml`
- transfer config: `experiments/variants/ensemble_transfer_musc_to_hsc.yaml`

The transfer config initializes from:
- `data/paper_runs/ensemble/ensemble_ref_musc/fold-1/checkpoints_baseline/M2--.ckpt`
- `data/paper_runs/ensemble/ensemble_ref_musc/fold-2/checkpoints_baseline/M2--.ckpt`

depending on the selected fold.

### Commands

Fold 1:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_ablation.py \
  --config experiments/variants/ensemble_transfer_musc_to_hsc.yaml \
  --fold 1
```

Fold 2:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_ablation.py \
  --config experiments/variants/ensemble_transfer_musc_to_hsc.yaml \
  --fold 2
```

---

## Phase 4: Compare Against HSC Fold Baselines

The correct comparison is:

- HSC baseline fold-1 vs MuSC->HSC transfer fold-1
- HSC baseline fold-2 vs MuSC->HSC transfer fold-2

Do not compare fold transfer runs against mixed rows.

Primary decision metric:
- test IoU

Secondary:
- test F1
- stability across both folds

Success criterion:
- transfer should improve at least one fold meaningfully without hurting the other badly
- ideally it should improve the fold average, not just one cherry-picked split

Failure criterion:
- if transfer is neutral on both folds, stop and do not expand the transfer branch further

---

## If MuSC Baseline Fails

If MuSC itself does not look promising:

1. stop transfer work
2. check crop-size adequacy first
3. only if crops are valid, consider whether MuSC is simply too different from HSC
4. then prefer a broader multi-dataset pretraining design over MuSC-only transfer

---

## Recommended Immediate Next Run Order

1. Branch the MuSC crop-size workflow away from `64`.
2. Generate MuSC QA crops/parquets for a larger crop size, starting with `128`.
3. Re-run `segmentation-size-stats` and a small manual crop review on the larger size.
4. If the larger crop size looks valid, run `ensemble_ref_musc` on fold-1 and fold-2.
5. Review MuSC fold results.
6. Only if MuSC looks reasonable at the validated crop size, run `ensemble_transfer_musc_to_hsc` on fold-1 and fold-2.

That is the shortest defensible path.
