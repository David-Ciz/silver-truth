# QA Model Data Preparation and Splitting Strategy

This document describes the current QA crop pipeline used in this repository.

## 1. QA Crop Creation

QA crops are generated from whole-image dataset dataframes using `silver-qa create-dataset`.

For each cell, the pipeline writes a stacked TIFF crop with shape `(4, H, W)`:

- `channel 0`: raw image crop
- `channel 1`: competitor segmentation crop (binary mask)
- `channel 2`: GT mask crop (binary mask)
- `channel 3`: TRA/tracking marker crop (binary mask)

Artifacts:

- Crop images directory:
  - `data/qa_crops/{DATASET}/sz{CROP_SIZE}/`
- Base QA parquet (no split column):
  - `data/dataframes/{DATASET}/qa_crops/base_sz{CROP_SIZE}_qa_dataset.parquet`

## 2. Split Attachment

Split assignment is performed with `silver-qa attach-split`.
It joins the base QA parquet against a whole-image dataframe (mixed/fold-1/fold-2)
and writes a split-specific QA parquet.

Outputs:

- `data/dataframes/{DATASET}/qa_crops/mixed_sz{CROP_SIZE}_qa_dataset.parquet`
- `data/dataframes/{DATASET}/qa_crops/fold-1_sz{CROP_SIZE}_qa_dataset.parquet`
- `data/dataframes/{DATASET}/qa_crops/fold-2_sz{CROP_SIZE}_qa_dataset.parquet`

## 3. Label Generation (`jaccard_score`)

CNN training requires a regression target (usually `jaccard_score`).
This is generated from QA crops via:

```bash
silver-evaluation calculate-evaluation-metrics-cli --mode cropped <qa_parquet>
```

This writes the following columns back into the same parquet:

- `jaccard_score`
- `f1_score`

In the DVC pipeline, this step is now run automatically in the split QA stages.

## 4. CNN Input Expectations

The QA CNN model consumes exactly 2 input channels.

- Default channels: `0,1` (`raw`, `segmentation`)
- Override with:
  - `silver-qa cnn train --input-channels "0,1" ...`
  - `silver-qa cnn evaluate --input-channels "0,1" ...`

This allows training from 4-channel crops while keeping model input fixed to two channels.

Use caution when changing channels:

- `channel 2` is GT and can leak target information into training.
- `channel 3` is tracking marker information and changes the modeling assumption.
