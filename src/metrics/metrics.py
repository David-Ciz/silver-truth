import numpy as np
from sklearn.metrics import jaccard_score


def calculate_jaccard_scores(gt_image, mask_image):
    labels = np.unique(gt_image)[1:]  # Exclude background (0)
    scores = {}
    for label in labels:
        label_layer = np.zeros_like(gt_image)
        label_layer[gt_image == label] = 1
        mask_layer = np.zeros_like(mask_image)
        mask_layer[mask_image == label] = 1
        j = jaccard_score(label_layer, mask_layer, average="micro")
        scores[label] = j
    return scores
