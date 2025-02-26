"""
This module contains the Label synchronization logic,
which is used to synchronize the labels and tracking data and in turn,
synchronize the labels between competitors, silver-truth, and the ground truth.
"""

import numpy as np
from scipy.spatial.distance import jaccard

from sqlalchemy.testing.plugin.plugin_base import logging


def verify_synchronization(label_img, tracking_img):
    """
    This function verifies the synchronization of the labels and tracking data.
    """
    # sanity checks
    if label_img is None or tracking_img is None:
        logging.warning("One of the images is missing.")
        return False
    try:
        label_img.shape == tracking_img.shape
    except AttributeError:
        logging.error("The images are not of the same shape.")
        return False

    tracking_uniques = np.unique(tracking_img)
    label_uniques = np.unique(label_img)

    if tracking_uniques == 0:
        logging.error("The tracking image is empty.")
        return False

    if label_uniques == 0:
        logging.warning("The label image is empty.")

    if label_uniques not in tracking_uniques:
        logging.error(
            "The label image contains labels that are not contained in tracking image."
        )
        return False

    for label in label_uniques:
        label_layer = label_img[label_img == label]
        tracking_layer = tracking_img[tracking_img == label]
        j_value = jaccard(label_layer, tracking_layer)
        if j_value == 0:
            logging.error(f"Jaccard index for label {label} is {j_value}.")
            return False

    return True
