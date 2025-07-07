# Jupyter Notebooks Overview

This document provides an overview of the Jupyter notebooks present in this repository. These notebooks serve various purposes, from supporting the core workflow to facilitating exploratory analysis, visualizations, and debugging.

## Categories

*   **Core Workflow Support**: Notebooks that directly support, verify, or are integral to the main data processing and evaluation pipeline.
*   **Exploratory & Analysis**: Notebooks used for deeper data analysis, understanding specific phenomena, or exploring new ideas and hypotheses.
*   **Visualization & Debugging**: Notebooks primarily focused on generating visual insights into data or processes, or for diagnosing specific issues.
*   **Historical/Archived**: Notebooks that are no longer relevant to the current project state and are kept purely for historical reference.

## Notebooks

### Core Workflow Support

*   **`finding_problem_labels.ipynb`**
    *   **Purpose**: Used to identify discrepancies between our synchronization algorithm and the one used in the competition, helping to understand slight differences in evaluation results.
    *   **Relevance**: Essential for understanding nuances in evaluation and synchronization.

*   **`verify_synchronization.ipynb`**
    *   **Purpose**: Used to verify the success and correctness of the label synchronization process.
    *   **Relevance**: Crucial for ensuring data quality after synchronization.

*   **`Exploring data.ipynb`**
    *   **Purpose**: Examines Ground Truth data, evaluating the number and size of segmentations. Useful for informing decisions on QA bounding box sizes.
    *   **Relevance**: Provides foundational insights into the dataset characteristics.

### Exploratory & Analysis

*   **`Artificial_dataset.ipynb`**
    *   **Purpose**: Used to create an artificial dataset, potentially for testing or demonstrating concepts.
    *   **Relevance**: Useful for generating synthetic data for specific experiments.

*   **`cherry_picking.ipynb`**
    *   **Purpose**: Contains parsed tables with cherry-picked evaluations for better combinations of input competitors, often used for specific analytical reports or dissertations.
    *   **Relevance**: For in-depth analysis of competitor combinations and their performance.

*   **`Fusion algorithm.ipynb`**
    *   **Purpose**: Likely explored or prototyped aspects of the fusion algorithm before its implementation in the main scripts.
    *   **Relevance**: Provides insights into the development and evolution of the fusion logic.

*   **`Image_preprocessing.ipynb`**
    *   **Purpose**: Explored various image preprocessing steps.
    *   **Relevance**: For understanding historical or alternative preprocessing approaches.

*   **`Preprocessing.ipynb`**
    *   **Purpose**: General data preprocessing explorations.
    *   **Relevance**: For understanding historical or alternative preprocessing approaches.

*   **`silver_truth_calculate_jaccards.ipynb`**
    *   **Purpose**: Used to calculate Jaccard indices for silver truth, potentially for ad-hoc analysis or validation.
    *   **Relevance**: For specific Jaccard calculations outside the main evaluation script.

### Visualization & Debugging

*   **`How-marker-looks.ipynb`**
    *   **Purpose**: Visualizes the tracking markers within the dataset.
    *   **Relevance**: For visual inspection and understanding of marker data.

*   **`image_picker_from_datasets.ipynb`**
    *   **Purpose**: Used to pick specific images from datasets, likely for visual inspection or creating subsets.
    *   **Relevance**: For targeted image retrieval and visualization.

*   **`label_graph_3.ipynb`**
    *   **Purpose**: Used to generate label graphs, potentially for visualizing relationships or tracking paths.
    *   **Relevance**: For advanced visualization of label structures.

*   **`label_in_source_visualizations.ipynb`**
    *   **Purpose**: Used to visualize labels directly within their source images.
    *   **Relevance**: For direct visual verification of label accuracy and placement.

*   **`presentation_graphics.ipynb`**
    *   **Purpose**: Used to generate graphics for presentations or reports.
    *   **Relevance**: For creating custom visualizations for communication.

*   **`visualizations_thumbnails.ipynb`**
    *   **Purpose**: Used to generate thumbnails for various visualizations.
    *   **Relevance**: For quick visual summaries of data or results.

*   **`Diagnosing_sync_bloat.ipynb`**
    *   **Purpose**: Used to diagnose issues related to "synchronization bloat."
    *   **Relevance**: For troubleshooting and understanding specific data synchronization problems.

### Historical/Archived

*(Currently, no notebooks are explicitly categorized as Historical/Archived, but this category is reserved for future use if notebooks become completely irrelevant.)*

## Important Note

The categorization above is based on the descriptions and the likely utility of these notebooks for various tasks within the project. It is always recommended to review the notebook content directly for the most accurate understanding of its functionality and applicability to your specific needs.