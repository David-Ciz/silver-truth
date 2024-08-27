import random

import numpy as np
from torch.utils.data import Dataset
import tifffile
import torch
from torchvision.transforms import functional as TF
from torchvision import transforms, datasets


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, -1]
        image = tifffile.imread(img_path)

        # Normalize the original image (grayscale values from 0-255 to 0-1)
        original_normalized = image[0] / 255.0

        # Stack the normalized image and binary mask
        combined = np.stack([original_normalized, image[1]], axis=2)

        label = self.dataframe.iloc[idx, -2]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image = self.transform(combined)

        return image, label


class RandomDiscreteRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


#def setup_transformations(min_width: int, min_height: int, mean: float, std: float, augment: bool):
def setup_transformations(augment: bool):
    # TODO: parametrize augmentations from a config file
    if augment:
        train_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            #transforms.Resize((min_width, min_height)),
            #RandomDiscreteRotation([90., 180., 270.]),  # Rotate by one of these degrees
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(23, padding=4),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            #transforms.Normalize(mean=mean, std=std)
    ])
    else:
        train_transform = transforms.Compose([
            #transforms.Resize((min_width, min_height)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=mean, std=std)
            ])
    # validation shouldn't have any augmentations that change the image, unless doing TTA
    val_transform = transforms.Compose([
        #transforms.Resize((min_width, min_height)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, val_transform