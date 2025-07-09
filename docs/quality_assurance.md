# Quality Assurance (QA)

This section outlines a proposed methodology for implementing a Quality Assurance (QA) step into the silver-truth generation pipeline. The goal of the QA process is to identify and potentially exclude low-quality segmentations from the fusion process, thereby improving the final fused output.

## QA Model

We propose a model that takes a raw source image and a competitor's segmentation as input and predicts the quality of the segmentation. This model can be implemented in two ways:

1.  **Classification Model:** Outputs a binary classification of "good" or "bad" for a given segmentation.
2.  **Regression Model:** Outputs a continuous value, such as the Jaccard Index, representing the quality of the segmentation.

This QA model can be applied at two different granularities:

*   **Image Level:** The model assesses the overall quality of a segmentation for an entire image.
*   **Cell Level:** The model assesses the quality of the segmentation for individual cells within an image.

## Experimental Framework

To evaluate the effectiveness of the QA model, we will test three different strategies for incorporating its output into the fusion pipeline. These strategies involve excluding data at different levels of granularity based on the QA model's predictions and a variety of confidence thresholds.

### 1. Competitor-Level Exclusion

In this approach, if a competitor's segmentations are consistently rated as "bad" by the QA model across multiple images, the entire competitor is excluded from the fusion process.

### 2. Image-Level Exclusion

This strategy involves excluding specific images from a competitor. If the QA model flags a particular segmentation from a competitor on a specific image as "bad", that single image segmentation is excluded from the fusion process, while the competitor's other segmentations are still used.

### 3. Cell-Level Exclusion

This is the most granular approach. The QA model identifies individual cell segmentations within an image that are of low quality. These specific "bad" cells are then removed from the fusion process, while the rest of the cells from that competitor's segmentation are retained.

By experimenting with these three strategies and adjusting the thresholds for what constitutes a "bad" segmentation, we can determine the most effective way to leverage the QA model to improve the quality of the final silver-truth annotations.

## Limitations and Future Work

It is important to note that this initial proposal is a first, naive approach. It has several limitations that should be addressed in future work:

*   **Error Types:** The current model does not explicitly differentiate between different types of errors, such as false positives and false negatives. A more sophisticated model would be able to identify not just *that* a segmentation is bad, but *why* (e.g., missing a cell vs. hallucinating a cell).
*   **Temporal Information:** The proposed model is static and does not incorporate temporal information from video sequences. Future iterations should leverage temporal consistency to improve the accuracy of the QA process. For example, a cell that is correctly identified in preceding and succeeding frames is more likely to be a true positive.
