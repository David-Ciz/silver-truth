# Architecture & Gotchas
_Things that are non-obvious and require digging through the source code to understand._

---

## How the Two-Stage Pipeline Works

```
Raw images + GT labels
      ↓
[Preprocessing] → Whole-image parquets (per competitor, with split column)
      ↓
[QA Crop Generation] → 64×64 crops: (raw, competitor_mask, gt_mask, tracking_marker)
      ↓
[QA Model] → ResNet50 regression → predicted_jaccard_index per crop
      ↓
[Filtering] → Threshold gate keeps "good" crops (+ top-1 fallback for uncovered cells)
      ↓
[Fusion] → Java voting (SIMPLE/MAJORITY/THRESHOLD/BIC) OR learned ensemble (UNet++)
      ↓
[Reconstruction] → Crops placed back into full images for evaluation
      ↓
[Evaluation] → Sparse-GT label-by-label IoU/F1
```

---

## Critical Architectural Details

### What "competitors" are

The 5 competitors are **existing cell segmentation methods** from the Cell Tracking Challenge that all ran on the same dataset:
- `CALT-US`, `DREX-US`, `KIT-Sch-GE`, `KTH-SE`, `MU-Lux-CZ`

Each produces a segmentation mask for the same cells. The pipeline's job is to combine (fuse) their outputs into a better segmentation than any individual competitor.

### How crops work

Each cell generates **~5-6 rows** in the QA parquet (one per competitor). All share the same `(campaign_number, original_image_key, label)` key but have different `competitor` and `stacked_path` columns. The actual "cell count" is the number of unique keys, NOT the parquet row count.

### The ensemble is a fusion method, not a separate model

The ensemble (`silver-ensemble`) is an alternative to Java voting. It takes competitor masks as input channels and learns to output a fused mask. It competes directly with Java fusion strategies like `SIMPLE` voting.

### Why the QA model uses 2 channels

The `--input-channels 0,1` flag selects channels from the 4-channel crop TIFF. Channel 0 is the raw image, channel 1 is the competitor mask. This means the QA model sees both the microscopy image and the competitor's segmentation attempt, and predicts how good that segmentation is (IoU vs ground truth).

### How full-image evaluation works

Crop-level IoU is **not** used in the paper. Crops clip cells at edges, distorting metrics.

The evaluation path:
1. Place each 64×64 crop back into the full image at its original coordinates
2. Use sparse-GT semantics: only score labels present in the ground truth
3. Compute per-label IoU, then aggregate

This is handled by `reconstruct_full_images_from_paths` (Java fusion) or `reconstruct_full_images_from_arrays` (ensemble).

---

## Gotchas

### Column name mismatch between fusion and reconstruction

`run-fusion-crops` outputs parquets with `crop_y_start/end/x_start/x_end` columns. But `reconstruct_full_images_from_paths` expects `recon_crop_*` columns. The `evaluate-fusion-crops` CLI handles the rename on-the-fly. This is not a bug — just confusing if you try to call reconstruction manually.

### `params.yaml` drives DVC crop stages

The DVC `foreach: ${datasets}` stages reference `params.yaml`. If a dataset is missing there, the QA crop stages will not reproduce for it. `BF-C2DL-MuSC` is now included, but keep this dependency in mind whenever adding another dataset.

### `attach-split` is "deprecated" but required

The docstring for `silver-qa attach-split` says DEPRECATED, but it is actively used in `dvc.yaml` QA crop split stages. Don't remove it.

### Mixed split has temporal bleed

The `mixed` split mode distributes GT images across train/val/test by cell count balancing **within a sequence**. Since microscopy sequences are temporally correlated (cells in frame T look similar to frame T+1), models trained on mixed splits see near-duplicates of their test data. **Do not use mixed-split results for paper claims.**

### Fold train/validation size must be audited at the supervised unit

Sparse GT means raw frame counts are not enough. A fold can look balanced by raw
frames while being badly imbalanced by GT cells, QA crop rows, or ensemble
databank rows.

Use the current preflight audit before training:

- [Experiment Preflight Methodology](experiment_preflight_methodology_2026-05-05.md)
- [Dataset Split Fix Plan](dataset_split_fix_plan_2026-05-05.md)

### Ensemble encoder is untrained (fix available)

Models in `models.py` defaulted to `encoder_weights=None`. The ResNet34
backbone was trained from scratch on tiny fold datasets. `--encoder-weights
imagenet` is available via CLI, but its effect must be re-evaluated under the
current split/crop protocol before making a result claim.

### Ensemble input channels (C1 vs C2)

| Version | Input | Channels | Notes |
|---|---|---:|---|
| `C1` (default) | Normalized competitor overlap | 1 | Pixel = fraction of competitors that agree |
| `C2` | Overlap + raw microscopy image | 2 | Implemented; old result claims were archived |

`C2` gives the ensemble access to what the cell actually looks like. Whether it
helps should be retested as an ensemble variant after the corrected split/crop
preflight passes.

### QA threshold is a validation-selected operating point

Archived runs suggested the hard QA threshold was brittle. Do not carry any old
threshold value forward as a result. Sweep thresholds, select the operating
point from validation only, and then evaluate the frozen choice on test.

### Java fusion can crash with `return code -6`

Known issue logged in `WORK_TRACKER.md`. The chunking/fallback mechanism in `run-fusion-crops` handles this, but standalone `run-fusion` calls may still fail.

---

## Helper Scripts

| Script | Purpose | Run time |
|---|---|---|
| `scripts/run_fold_comparison.sh` | Train + evaluate ensemble vs competitors on both folds. Fresh data each run (timestamped output). | ~10-15 min |
| `scripts/run_fair_comparison.sh` | Evaluate CALT-US through the same crop-reconstruct path used by the ensemble — apples-to-apples. | ~5 min |
| `scripts/run_experiment_battery.sh` | Full experiment battery: 4 configs (baseline, strong_aug, c2_raw, c2_strong) × 2 folds = 8 runs. Auto-compiles results. | ~40 min |
| `scripts/compile_results.py` | Reads experiment parquets/CSVs, produces a markdown summary with cross-fold averages vs CALT-US. Can run standalone on any results dir. | instant |
| `scripts/test_pretrained_ensemble.sh` | A/B test: pretrained (ImageNet) vs scratch encoder on fold-1. | ~5 min |
| `scripts/run_ablation.py` | Full ablation runner for all phases (competitor, fusion, QA, ensemble, threshold sweeps). | ~60 min per split |

---

## Archived Result Tables

Older result tables were removed from this active architecture note because the
underlying split/crop inputs were invalidated by the May 2026 audit. Treat their
ideas as rerun hypotheses, not as evidence.

See:

- [Clean Slate Experiment Rerun Plan](clean_slate_experiment_rerun_plan_2026-05-05.md)
- [Rerun Hypotheses Backlog](rerun_hypotheses_backlog_2026-05-05.md)

### Ensemble transfer path

The runner now supports initializing a new ensemble training run from an existing ensemble checkpoint:
- pretrain config: `experiments/variants/ensemble_ref_musc.yaml`
- fine-tune config: `experiments/variants/ensemble_transfer_musc_to_hsc.yaml`

Preferred usage is fold-locked:
- `BF-C2DL-MuSC fold-1 -> BF-C2DL-HSC fold-1`
- `BF-C2DL-MuSC fold-2 -> BF-C2DL-HSC fold-2`

This is the preferred next experiment path over more one-off scripts.

---

## Environment Setup

Must-do steps not fully obvious from the README:

```bash
# 1. Clone and set up
git clone https://github.com/David-Ciz/silver-truth
cd silver-truth
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# 2. MLflow (REQUIRED for any experiment command)
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/mlruns
mkdir -p data/mlflow/mlruns

# 3. Pull data (requires SSH access to Karolina HPC)
dvc remote add hpc_storage ssh://karolina.it4i.cz/mnt/proj1/eu-25-40/innovaite/dvc_store
dvc remote default hpc_storage
dvc pull

# 4. Verify DVC prep stages
dvc repro create_fold1@BF-C2DL-HSC create_fold2@BF-C2DL-HSC
```

**Important**: Always run commands from `.venv`. System Python will cause dependency mismatches, especially for `pytorch_lightning` and `segmentation_models_pytorch`.
