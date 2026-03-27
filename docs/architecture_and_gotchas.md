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

### Fold-2 has asymmetrically tiny training data

| | fold-1 | fold-2 | mixed |
|---|---:|---:|---:|
| Train cells | 139 | **48** | 394 |
| Train images | 36 | **6** | 16 |

Fold-2 trains on only 6 labeled images / 48 cells. This is why most learned models (QA, ensemble) perform dramatically worse on fold-2.

### Ensemble encoder is untrained (fix available)

Models in `models.py` defaulted to `encoder_weights=None`. The ResNet34 backbone was trained from scratch on tiny fold datasets. **Fix**: `--encoder-weights imagenet` is now available via CLI but testing showed it's roughly neutral (ImageNet features don't transfer well to binary mask inputs).

### Ensemble input channels (C1 vs C2)

| Version | Input | Channels | Notes |
|---|---|---:|---|
| `C1` (default) | Normalized competitor overlap | 1 | Pixel = fraction of competitors that agree |
| `C2` | Overlap + raw microscopy image | 2 | Implemented and tested; underperforms the current `C1` baseline |

`C2` gives the ensemble access to what the cell actually looks like, but the current experiments show it is not helping enough to justify using it in the main paper path.

### QA best threshold is 0.50, not 0.75

The filtering validity reports from both folds show that the best-F1 threshold is 0.50. The `t=0.75` currently used in the ablation removes too many crops, especially on fold-2 where 249/255 test cells are left with only 1 crop after filtering.

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

## Latest Results (2026-03-13)

### Fold Comparison: Ensemble vs CALT-US (test split, full-image IoU)

| Method | Fold-1 | Fold-2 | Average |
|---|---:|---:|---:|
| **CALT-US** | 0.8526 | 0.8953 | 0.8739 |
| **ENSEMBLE (ours)** | **0.8912** | 0.8787 | **0.8850** |
| Δ | **+0.0386** | −0.0166 | **+0.0110** |

### Key Observations
- Ensemble beats CALT-US on fold-1 by +0.0386 and on cross-fold average by +0.0110
- Fold-2 underperforms (only 48 training cells / 6 images)
- Pretrained encoder (ImageNet) ≈ scratch encoder (neutral result)
- QA filtering at t=0.50 is neutral; t=0.75 hurts slightly

> **Note**: Competitor IoU is computed on whole images; ensemble IoU goes through crop reconstruction. Use `run_fair_comparison.sh` for a direct apples-to-apples comparison.

### Ensemble Battery: C1 vs C2, basic vs strong augmentation

Results from `scripts/run_experiment_battery.sh` after fixing the C2/evaluation path:

| Experiment | Fold | IoU | F1 | N |
|---|---|---:|---:|---:|
| `baseline` | fold-1 | 0.8912 | 0.9423 | 8 |
| `baseline` | fold-2 | 0.8787 | 0.9350 | 49 |
| `c2_raw` | fold-1 | 0.8696 | 0.9301 | 8 |
| `c2_raw` | fold-2 | 0.8618 | 0.9252 | 49 |
| `c2_strong` | fold-1 | 0.8794 | 0.9357 | 8 |
| `c2_strong` | fold-2 | 0.8528 | 0.9199 | 49 |
| `strong_aug` | fold-1 | 0.8866 | 0.9398 | 8 |
| `strong_aug` | fold-2 | 0.8502 | 0.9183 | 49 |

Weighted summary (`N=57` total):

| Experiment | weighted IoU | weighted F1 |
|---|---:|---:|
| `baseline` | **0.8805** | **0.9360** |
| `c2_raw` | 0.8629 | 0.9259 |
| `c2_strong` | 0.8565 | 0.9221 |
| `strong_aug` | 0.8553 | 0.9213 |

Takeaway:
- keep `C1 + basic` as the default ensemble setup
- do not move `C2` into the main paper path without a separate rescue experiment
- the current `strong` augmentation preset is too aggressive for this data regime

### Focused ensemble sweep after the battery

Using `scripts/run_ablation.py` with `experiments/variants/ensemble_*.yaml`:

| Variant | Split | IoU | F1 | N |
|---|---|---:|---:|---:|
| `ensemble_ref` | mixed | 0.9054 | 0.9500 | 29 |
| `ensemble_aug_vflip_brightness` | fold-2 | 0.8785 | 0.9349 | 49 |
| `ensemble_aug_brightness` | fold-2 | 0.8784 | 0.9348 | 49 |
| `ensemble_imagenet` | fold-2 | 0.8783 | 0.9349 | 49 |
| `ensemble_unet` | fold-2 | 0.8708 | 0.9305 | 49 |
| `ensemble_aug_vflip` | fold-2 | 0.8684 | 0.9291 | 49 |
| `ensemble_aug_noise` | fold-2 | 0.8271 | 0.9042 | 49 |

Takeaway:
- no fold-2 sweep variant clearly beats the current baseline (`0.8787`)
- `ImageNet` encoder init remains roughly neutral
- `GaussianNoise` is harmful and should be dropped
- the next meaningful step is dataset transfer, not more local augmentation search

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
