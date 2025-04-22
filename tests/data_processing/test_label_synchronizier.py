import unittest
import numpy as np
from src.data_processing.label_synchronizer import verify_synchronization

class TestVerifySynchronization(unittest.TestCase):

    def test_both_images_none(self):
        self.assertFalse(verify_synchronization(None, None))

    def test_one_image_none(self):
        label_img = np.array([[1, 2], [3, 4]])
        self.assertFalse(verify_synchronization(label_img, None))
        self.assertFalse(verify_synchronization(None, label_img))

    def test_different_shapes(self):
        label_img = np.array([[1, 2], [3, 4]])
        tracking_img = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(verify_synchronization(label_img, tracking_img))

    def test_empty_tracking_image(self):
        label_img = np.array([[1, 2], [3, 4]])
        tracking_img = np.array([[0, 0], [0, 0]])
        self.assertFalse(verify_synchronization(label_img, tracking_img))

    def test_empty_label_image(self):
        label_img = np.array([[0, 0], [0, 0]])
        tracking_img = np.array([[1, 2], [3, 4]])
        self.assertTrue(verify_synchronization(label_img, tracking_img))

    def test_synchronized_images(self):
        label_img = np.array([[1, 2], [3, 4]])
        tracking_img = np.array([[1, 2], [3, 4]])
        self.assertTrue(verify_synchronization(label_img, tracking_img))

    def test_not_synchronized_images(self):
        label_img = np.array([[1, 2], [3, 4]])
        tracking_img = np.array([[1, 2], [4, 3]])
        self.assertFalse(verify_synchronization(label_img, tracking_img))

if __name__ == '__main__':
    unittest.main()