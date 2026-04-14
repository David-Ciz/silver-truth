# Ablation Results And Insights
_Updated: 2026-03-13 | Dataset: BF-C2DL-HSC_

This note separates the decision-relevant readout from the operational plan in `docs/ablation_experiment_plan.md`.

Recent QA follow-up runs after this note are summarized separately in
`docs/qa_follow_up_experiments_2026-03-27.md`.

Current status:
- the old fold result folders were deleted
- the fold rerun is in progress
- the mixed-run comparison surface is still available in MLflow and is the best current signal for where to focus next
- MuSC -> HSC ensemble transfer has now been run and improved the HSC ensemble baseline on both folds

---

## Experiment Battery (2026-03-13)

The 4-way ensemble battery (`baseline`, `strong_aug`, `c2_raw`, `c2_strong`) was rerun on both folds with the repaired C2 path.

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

Weighted by the actual test counts (`N=57` total):

| Experiment | weighted IoU | weighted F1 |
|---|---:|---:|
| `baseline` | **0.8805** | **0.9360** |
| `c2_raw` | 0.8629 | 0.9259 |
| `c2_strong` | 0.8565 | 0.9221 |
| `strong_aug` | 0.8553 | 0.9213 |

Decision-level readout:
- `baseline` is the clear winner on both folds and on the weighted aggregate
- adding the raw microscopy channel (`C2`) hurts rather than helps in the current setup
- the current `strong` augmentation preset hurts even without the raw channel
- `c2_strong` does not recover the C2 loss, so the issue is not "C2 needs stronger augmentation"

What this means operationally:
- keep `C1 + basic augmentation` as the default ensemble configuration
- drop `C2` from the main paper path for now
- do not use the current `strong` augmentation preset as a default or paper candidate
- treat any further C2 work as a targeted rescue attempt, not a parallel mainline

### Focused Ensemble Sweep (2026-03-13)

A smaller config-driven sweep was then run through `scripts/run_ablation.py` using the new `experiments/variants/ensemble_*.yaml` files.

| Variant | Split | IoU | F1 | N | Readout |
|---|---|---:|---:|---:|---|
| `ensemble_ref` | mixed | **0.9054** | **0.9500** | 29 | strong mixed reference |
| `ensemble_aug_vflip_brightness` | fold-2 | 0.8785 | 0.9349 | 49 | effectively tied with baseline |
| `ensemble_aug_brightness` | fold-2 | 0.8784 | 0.9348 | 49 | effectively tied with baseline |
| `ensemble_imagenet` | fold-2 | 0.8783 | 0.9349 | 49 | effectively tied with baseline |
| `ensemble_unet` | fold-2 | 0.8708 | 0.9305 | 49 | worse than U-Net++ |
| `ensemble_aug_vflip` | fold-2 | 0.8684 | 0.9291 | 49 | worse than baseline |
| `ensemble_aug_noise` | fold-2 | 0.8271 | 0.9042 | 49 | clearly harmful |

Interpretation:
- none of the targeted fold-2 variants clearly beats the current fold baseline (`0.8787`)
- `ImageNet` initialization is neutral at best, not a meaningful gain
- `Unet` should be dropped from the immediate search space
- `GaussianNoise` should be dropped entirely for this dataset/setup
- if a promotion round is needed, only `ensemble_imagenet` and `ensemble_aug_vflip_brightness` are worth taking to fold-1

This narrows the next real lever further:
- stop local augmentation tinkering after the promotion round
- move to dataset-level transfer rather than more C1/C2 or augmentation branching

---

## What The Fold Runs Said

These numbers are preserved from the completed fold run summary that was already written into the plan before the result folders were deleted.

| Row | fold-1 | fold-2 | average |
|---|---:|---:|---:|
| Best competitor (`CALT-US`) | 0.8526 | 0.8953 | 0.8739 |
| `fusion_only / SIMPLE` | 0.8444 | 0.8875 | 0.8659 |
| `ensemble_only` | 0.8469 | 0.8433 | 0.8451 |
| `full_pipeline / SIMPLE` (`t=0.75`) | 0.8280 | 0.8007 | 0.8144 |
| `ensemble_qa` (`t=0.75`) | 0.8354 | 0.7931 | 0.8142 |
| `ensemble_qa_retrained` (`t=0.75`) | 0.8401 | 0.7985 | 0.8193 |
| `qa_only / top-1` | 0.8522 | 0.7950 | 0.8236 |

Main readout:
- on fold-based runs, the pipeline does not beat the best individual competitor
- QA hurts more than it helps in the current form
- the ensemble is respectable, but not enough to justify the extra moving parts if the target is fold-based performance alone

---

## What The Mixed Run Says

These numbers come from `paper-compare-BF-C2DL-HSC-baseline-mixed` in MLflow.

### Strongest current rows

| Row | evaluation level | test IoU | test F1 |
|---|---|---:|---:|
| `ensemble_only` | image reconstructed | 0.9054 | 0.9500 |
| `ensemble_qa_retrained_t0.75` | image reconstructed | 0.9053 | 0.9499 |
| `ensemble_qa_t0.50` | image reconstructed | 0.9054 | 0.9500 |
| `ensemble_qa_t0.60` | image reconstructed | 0.9044 | 0.9494 |
| `ensemble_qa_t0.70` | image reconstructed | 0.9031 | 0.9487 |
| `ensemble_qa_t0.75` | image reconstructed | 0.9028 | 0.9486 |
| `full_pipeline_t0.70__simple` | image reconstructed | 0.9018 | 0.9480 |
| `full_pipeline_t0.75__simple` | image reconstructed | 0.9011 | 0.9476 |
| `fusion_only__simple` | image reconstructed | 0.8919 | 0.9421 |
| `SILVER-TRUTH` | full-image label | 0.8876 | 0.9391 |
| best competitor (`CALT-US`) | full-image label | 0.8830 | 0.9372 |

### Mixed-run interpretation

- this is the first setup where the learned ensemble clearly becomes the strongest row
- the gain is real, but it is mostly a data effect, not a QA effect
- QA gating does not improve the ensemble in a meaningful way on mixed data
- retraining on the QA-filtered databank also does not move the needle in a useful way
- `SIMPLE` fusion does benefit a little from QA on mixed, but it still trails the ensemble

The most telling comparison is simple:

| Row | test IoU |
|---|---:|
| best competitor (`CALT-US`) | 0.8830 |
| `SILVER-TRUTH` | 0.8876 |
| `fusion_only__simple` | 0.8919 |
| `full_pipeline_t0.70__simple` | 0.9018 |
| `ensemble_only` | 0.9054 |

That pattern is hard to explain as an evaluation bug. It looks much more like a data regime issue:
- with mixed data, the ensemble finally has enough variation to learn something useful
- with fold-only training, it does not

---

## The Practical Takeaway

If the paper story has to stay grounded in what currently works, the cleanest direction is:

1. Treat the ensemble as the main model branch.
2. Treat mixed training as the main data regime.
3. Treat QA as optional analysis, not as the core mechanism.

Right now the strongest statement supported by the results is not:

> "QA filtering improves the full pipeline."

It is closer to:

> "A learned fusion model becomes competitive or best once it is trained on enough labeled cells, while the current QA stage adds little."

---

## Suggested Next Experiments

### 0. Freeze the current default

Before running more sweeps, lock this in as the reference:
- ensemble default = `C1`
- augmentation default = `basic`
- benchmark row = `baseline`

Every new experiment should be compared directly against that row, not against the weaker C2 or strong-aug variants.

### 1. Put the next effort into data, not QA

The current results do not justify spending more iteration budget on QA-first ideas. The ensemble mixed run already tells the more useful story.

Priority:
- keep QA out of the critical path
- keep the unfiltered ensemble as the main baseline
- only retain QA as a secondary sweep if it is cheap to run

### 2. Try a smaller, controlled augmentation sweep instead of `strong`

The current `strong` preset is too blunt. The next useful question is not "strong or not", but which single augmentations help.

Recommended mini-sweep:
- `baseline` (control)
- `+VerticalFlip`
- `+BrightnessContrast` only
- `+GaussianNoise` only
- `+VerticalFlip + BrightnessContrast`

Stop rule:
- if none of these beats baseline on fold-2, stop augment tuning and move on

### 3. Pretrain the ensemble on similar datasets

This is the most natural next step if the logic looks sound and the bottleneck is sample count.

Recommended order:
- first try pretraining on `BF-C2DL-MuSC`
- then fine-tune on `BF-C2DL-HSC` mixed
- only after that consider broader multi-dataset pretraining

Why this order:
- it stays within a similar brightfield cell-tracking regime first
- it reduces the chance that a large modality gap dominates the training signal

Current implementation status:
- `ImageNet` encoder init already exists and appears neutral
- true ensemble transfer is now supported via checkpoint initialization
- config path:
  - pretrain config: `experiments/variants/ensemble_ref_musc.yaml`
  - fine-tune config: `experiments/variants/ensemble_transfer_musc_to_hsc.yaml`

Recommended paper-safe transfer run order:
1. pretrain on `BF-C2DL-MuSC` fold-1
2. fine-tune that checkpoint on `BF-C2DL-HSC` fold-1
3. pretrain on `BF-C2DL-MuSC` fold-2
4. fine-tune that checkpoint on `BF-C2DL-HSC` fold-2

### 3a. MuSC -> HSC transfer result

This experiment has now been completed using the MuSC `sz256` ensemble checkpoint
as initialization for the HSC `sz64` ensemble.

Important semantics:

- this was a **MuSC-pretrained, HSC-fine-tuned** ensemble run
- it was not a direct cold application of the MuSC model to HSC
- the source weights were reused, then the target model was retrained on HSC
- this distinction matters because the later QA zero-shot transfer result is negative

Fair comparison:

- scratch HSC ensemble baseline vs transfer HSC ensemble baseline
- same final evaluation level: `full_image_label`
- metric: `test_iou`

| Fold | HSC scratch ensemble | HSC transfer ensemble | Delta |
|---|---:|---:|---:|
| `fold-1` | `0.8562` | `0.8718` | `+0.0156` |
| `fold-2` | `0.8519` | `0.8699` | `+0.0180` |
| mean | `0.8540` | `0.8708` | `+0.0168` |

Interpretation:

- transfer helped on both HSC folds
- the gain is larger than the "noise floor" of earlier local ensemble tweaks
- this is the first ensemble-only change in the current branch that clearly improves the fold baseline without depending on QA

Important limit:

- this still does not beat HSC silver truth overall
- but it does materially improve the learned ensemble branch, which makes transfer a justified next-step story in the paper
5. compare against the matching HSC fold baselines

### 4. Only then try a larger pooled training set

Combining all Cell Tracking Challenge datasets into one pool may help, but it is not the first thing I would try blindly.

Better framing:
- use all-dataset training as pretraining, not as the final target regime
- fine-tune on the target dataset afterward
- sample by dataset carefully so one large dataset does not dominate

If you do try it, make it a controlled comparison:
- target-only mixed
- similar-dataset pretrain -> target mixed fine-tune
- all-dataset pretrain -> target mixed fine-tune

### 5. Match architecture size to the actual sample count

The fold splits are much smaller than they look if you count real cells instead of crop rows.

Implication:
- fold-only runs are too small to justify large-capacity models trained from scratch
- mixed is still not large, but it is much more plausible

This argues for:
- smaller backbones
- stronger regularization
- transfer learning or pretraining
- frozen or partially frozen encoders during fine-tuning

### 6. Run one narrow architecture check before broad search

Because `baseline` already beats the other battery branches, the next architecture step should stay narrow:
- `UnetPlusPlus` baseline (control)
- `Unet` with same `C1/basic` setup
- `UnetPlusPlus` with ImageNet encoder weights

This is enough to tell whether gains are more likely to come from architecture/initialization than from feature changes like `C2`.

### 7. Use the mixed ensemble as the benchmark for every new idea

For the next round, the default question should be:

> "Does this beat `ensemble_only` on mixed?"

If not, it probably does not belong in the main paper path.

---

## My Current Recommendation

If you want one concrete next move rather than a menu:

1. Keep `C1 + basic augmentation` as the locked ensemble default.
2. Run a very small augmentation ablation around that baseline, not the current `strong` preset.
3. Keep `ensemble_only` on mixed as the main reference row.
4. Run a fold-locked transfer experiment:
   `BF-C2DL-MuSC fold-k -> BF-C2DL-HSC fold-k`
5. Compare that directly against the current HSC fold baseline for the same fold.

That is the most plausible route to a real improvement without inventing a new pipeline story.
