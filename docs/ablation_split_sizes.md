# Ablation Split Sizes
_Updated: 2026-03-12 | Dataset: BF-C2DL-HSC_

This note records how much supervised data is actually available to the pipeline in each split.

Why this matters:
- the whole-image parquet can look large because it contains many unlabeled frames
- the QA parquet can look large because it contains one row per candidate crop
- the ensemble databank for `C1` is effectively built around one fused training example per real cell, not per competitor row

So the important count is:

> actual labeled cells = unique `(campaign_number, original_image_key, label)`

That is the number that should drive architecture choices.

---

## Whole-Image Supervision

These counts come from the whole-image split parquets and show how many image rows actually carry GT.

| Split setup | train labeled images | validation labeled images | test labeled images | total labeled images |
|---|---:|---:|---:|---:|
| fold-1 | 36 | 13 | 8 | 57 |
| fold-2 | 6 | 2 | 49 | 57 |
| mixed | 16 | 12 | 29 | 57 |

The fold asymmetry is severe:
- fold-1 trains on 36 labeled images
- fold-2 trains on only 6 labeled images

That already explains why fold-2 is such a hostile regime for a learned fusion model.

---

## Actual Labeled Cells

These counts come from the QA crop parquets after collapsing competitor-specific rows down to real cells.

| Split setup | train cells | validation cells | test cells | train+val cells |
|---|---:|---:|---:|---:|
| fold-1 | 139 | 116 | 309 | 255 |
| fold-2 | 48 | 261 | 255 | 309 |
| mixed | 394 | 84 | 86 | 478 |

This is the number that matters most for the ensemble.

Two points stand out:
- fold-2 has only 48 real training cells
- mixed has 394 real training cells, almost 3x fold-1 and more than 8x fold-2

That lines up very well with the empirical result that the ensemble only becomes clearly strong on mixed training.

---

## Candidate Crop Rows

For completeness, these are the raw QA parquet row counts before collapsing over competitors.

| Split setup | train rows | validation rows | test rows | total rows |
|---|---:|---:|---:|---:|
| fold-1 | 824 | 685 | 1809 | 3318 |
| fold-2 | 270 | 1539 | 1509 | 3318 |
| mixed | 2315 | 492 | 511 | 3318 |

Average competitors per actual cell:
- fold-1: about `5.9`
- fold-2: about `5.9`
- mixed: about `5.9`

So the row count is mostly "number of cells x number of available competitor crops", not independent supervision.

---

## What This Means For Architecture Choice

### Fold-only training

The fold splits are extremely small for learning-heavy models:
- fold-1 gives 139 train cells
- fold-2 gives 48 train cells

That is a transfer-learning regime, not a comfortable train-from-scratch regime.

Practical implication:
- smaller models are easier to justify
- pretrained encoders make more sense than deeper custom models
- regularization matters more than capacity

### Mixed training

Mixed is still small, but at least plausible:
- 394 train cells
- 478 train+val cells

That is enough to explain why the ensemble starts working there while the fold runs stay unstable or underpowered.

### What I would not assume

I would not assume that a worse fold result means the logic is broken.

Given these counts, a simpler explanation is:
- the fold splits just do not provide enough labeled cells for the ensemble to learn a robust fusion rule

---

## Bottom Line

If the goal is to decide what models are realistic:

- fold-2 especially is too small for ambitious from-scratch architectures
- fold-1 is still small
- mixed is the only currently convincing regime for model development

That makes pretraining, transfer, or multi-dataset training the natural next step rather than more complexity in the QA branch.
