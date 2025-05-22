# Cell Tracking Challenge Dataset Structure

## Overview

The project contains 16 distinct biological datasets from the Cell Tracking Challenge, each featuring different cell types and imaging modalities. Each dataset follows a standardized structure with multiple annotations and results.

## Dataset Categories

| Dataset ID | Description | Modality |
|------------|-------------|----------|
| BF-C2DL-HSC | Hematopoietic Stem Cells | Brightfield 2D |
| BF-C2DL-MuSC | Muscle Stem Cells | Brightfield 2D |
| DIC-C2DH-HeLa | HeLa Cells | DIC 2D |
| Fluo-C2DL-MSC | Mesenchymal Stem Cells | Fluorescence 2D |
| Fluo-C3DH-A549 | Lung Cancer Cells | Fluorescence 3D |
| Fluo-C3DH-A549-SIM | Simulated Lung Cancer Cells | Fluorescence 3D |
| Fluo-C3DH-H157 | Lung Cancer Cells | Fluorescence 3D |
| Fluo-C3DL-MDA231 | Breast Cancer Cells | Fluorescence 3D |
| Fluo-N2DH-GOWT1 | Embryonic Stem Cells | Fluorescence 2D |
| Fluo-N2DH-SIM+ | Simulated Nuclei | Fluorescence 2D |
| Fluo-N2DL-HeLa | HeLa Cell Nuclei | Fluorescence 2D |
| Fluo-N3DH-CE | C. elegans Embryo Nuclei | Fluorescence 3D |
| Fluo-N3DH-CHO | Chinese Hamster Ovary Nuclei | Fluorescence 3D |
| Fluo-N3DH-SIM+ | Simulated Nuclei | Fluorescence 3D |
| PhC-C2DH-U373 | Glioblastoma-astrocytoma Cells | Phase Contrast 2D |
| PhC-C2DL-PSC | Pancreatic Stem Cells | Phase Contrast 2D |

## Directory Structure

Each dataset follows this structure:
<pre>
Dataset/

├── 01/                 # First sequence raw images

├── 01_GT/             # Ground truth for first sequence

│   ├── SEG/           # Segmentation ground truth (sparse)

│   └── TRA/           # Tracking ground truth (point annotations)

├── 01_GT_sync/        # Synchronized ground truth

├── 01_ST/             # Silver truth for first sequence

├── 01_ST_sync/        # Synchronized silver truth

├── 02/                # Second sequence raw images

├── 02_GT/             # Ground truth for second sequence

│   ├── SEG/           # Segmentation ground truth (sparse)

│   └── TRA/           # Tracking ground truth (point annotations)

├── 02_GT_sync/        # Synchronized ground truth

├── 02_ST/             # Silver truth for second sequence

├── 02_ST_sync/        # Synchronized silver truth

└── Competitors/       # Competitor results

├── CALT-US/

│   ├── 01_RES/    # Results for first sequence

│   └── 02_RES/    # Results for second sequence

├── DREX-US/

├── KIT-Sch-GE/

├── KTH-SE/

└── MU-Lux-CZ/
</pre>
## Data Types

### Raw Images (01/ and 02/)
- Format: TIFF files
- Content: Original microscopy images
- Two independent sequences per dataset

### Ground Truth (01_GT/ and 02_GT/)
Contains two types of annotations:

1. **SEG/ (Segmentation)**
   - Sparse manual segmentations by human experts
   - TIFF format
   - Full cell masks at selected time points
   - Used for segmentation accuracy evaluation

2. **TRA/ (Tracking)**
   - Point annotations for cell tracking
   - TIFF format
   - Single point per cell in each frame
   - Used for tracking accuracy evaluation

### Silver Truth (01_ST/ and 02_ST/)
- Combined best results from competition participants
- Generated through consensus of top-performing methods
- TIFF format
- Provides more dense annotations than ground truth

### Synchronized Versions (*_sync/)
- Processed versions of GT and ST
- Ensures consistent labeling across temporal sequences
- Maintains label correspondence between frames
- TIFF format

### Competitor Results
Each competitor folder contains:
- 01_RES/: Results for first sequence
- 02_RES/: Results for second sequence
- All results in TIFF format
- Each competitor uses their own methodology

## File Format Specifications

All images are stored in TIFF format with the following characteristics:
- Segmentation masks: Binary or labeled images where each cell has a unique identifier
- Tracking markers: Binary images with single points marking cell positions
- Raw images: Original microscopy data in the respective modality

## Usage Notes

- When working with the dataset, always maintain the original directory structure
- Synchronization between SEG and TRA files is crucial for accurate evaluation
- Silver truth can be used as a more extensive ground truth alternative
- Each sequence (01 and 02) should be processed independently

## Related Documentation

- [Label Synchronization Process](../data_processing/label_synchronization.md)
- [Evaluation Metrics](../evaluation/metrics.md)
