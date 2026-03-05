from typing import Callable, Optional
import tifffile
from tqdm import tqdm
from enum import Enum
from torch.utils.data import Dataset
import src.silver_truth.ensemble.external as ext
import albumentations as A
import numpy as np
import time
# from PIL import Image


class Version(Enum):
    A1 = 1  # raw, gt           [1,1]
    B1 = 3  # seg, gt           [1,1]
    B2 = 4  # seg&raw, gt       [2,1]
    B3 = 5  # segs, gt          [N,1]
    C1 = 6  # norm_seg, gt      [1,1]
    C2 = 7  # norm_seg&raw, gt  [2,1]


def get_dataset_class(version: Version):
    if version == Version.A1:
        return EnsembleDatasetA1
    elif version == Version.B1:
        return EnsembleDatasetB1
    elif version == Version.B3:
        return EnsembleDatasetB3
    elif version == Version.C1:
        return EnsembleDatasetC1
    elif version == Version.C2:
        return EnsembleDatasetC2
    else:
        raise Exception(f"Error: dataset version '{version.name}' not implemented!")


class EnsembleDatasetA1(Dataset):
    """
    Ensemble dataset data structure A1.

    Input: crop image with the raw image.
    Label: crop image of ground truth.
    """

    def __init__(
        self,
        ensemble_parquet_path,
        split,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        # load dataframe
        df = ext.load_parquet(ensemble_parquet_path)

        self.version = Version.A1
        if transform is None:
            self.transform = A.Compose([A.ToTensorV2()])
        else:
            self.transform = transform
        self.data = []
        self.gts = []

        # fill tensors with actual data
        for index, row in enumerate(df.itertuples()):
            if split != "all":
                if row.split != split:
                    continue
            # load the image
            composed_image = tifffile.imread(row.image_path)  # type: ignore
            # split the composed image

            # albumentations require (H, W, C) for images
            raw_image = (
                composed_image[2].astype(dtype=np.float32) / 255
            )  # scale down to [0,1]
            gt_image = (
                composed_image[1].astype(dtype=np.float32) / 255
            )  # scale down to [0,1]
            self.data.append(raw_image)
            self.gts.append(gt_image)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # return self.transform(self.data[index]), self.transform(self.gts[index])
        augmented = self.transform(image=self.data[index], mask=self.gts[index])
        return augmented["image"], augmented["mask"].unsqueeze(-3)


class EnsembleDatasetB1(Dataset):
    """
    Ensemble dataset data structure B1.

    Input: crop image with competitor segmentation.
    Label: crop image of ground truth.
    """

    def __init__(
        self,
        ensemble_parquet_path,
        split,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        # load dataframe
        df = ext.load_parquet(ensemble_parquet_path)

        self.version = Version.B1
        if transform is None:
            self.transform = A.Compose([A.ToTensorV2()])
        else:
            self.transform = transform
        self.data = []
        self.gts = []

        # fill tensors with actual data
        for index, row in enumerate(df.itertuples()):
            if split != "all":
                if row.split != split:
                    continue
            # load the image
            composed_image = tifffile.imread(row.image_path)  # type: ignore
            # split the composed image

            # albumentations require (H, W, C) for images
            segmentation = (
                composed_image[0].astype(dtype=np.float32) / 255
            )  # scale down to [0,1]
            gt_image = (
                composed_image[1].astype(dtype=np.float32) / 255
            )  # scale down to [0,1]
            self.data.append(segmentation)
            self.gts.append(gt_image)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # return self.transform(self.data[index]), self.transform(self.gts[index])
        augmented = self.transform(image=self.data[index], mask=self.gts[index])
        return augmented["image"], augmented["mask"].unsqueeze(-3)


class EnsembleDatasetB3(Dataset):
    """
    Ensemble dataset data structure B3.

    Input: crop images of competitors segmentations.
    Label: crop image of ground truth.
    """

    def __init__(
        self,
        ensemble_parquet_path,
        split,
        transform: Optional[Callable] = None,
        num_inputs: int = 8,
    ) -> None:
        super().__init__()

        # load dataframe
        df = ext.load_parquet(ensemble_parquet_path)

        self.version = Version.B3
        if transform is None:
            self.transform = A.Compose([A.ToTensorV2()])
        else:
            self.transform = transform
        self.data = []
        self.gts = []

        unique_cell_ids = df["full_cell_id"].unique()

        for full_cell_id in unique_cell_ids:
            df_same_cell = df[df["full_cell_id"] == full_cell_id]
            segmentations = []
            gt_image = ()
            # fill tensors with actual data
            for index, row in enumerate(df_same_cell.itertuples()):
                if split != "all":
                    if row.split != split:
                        continue
                # load the image
                composed_image = tifffile.imread(row.image_path)  # type: ignore
                # albumentations require (H, W, C) for images
                segmentations.append(
                    composed_image[0].astype(dtype=np.float32) / 255
                )  # scale down to [0,1]
                gt_image = (
                    composed_image[1].astype(dtype=np.float32) / 255
                )  # scale down to [0,1]

            if len(segmentations) > 0:
                h, w = segmentations[0].shape
                # add empty images to fill all input channels
                for _ in range(num_inputs - len(segmentations)):
                    segmentations.append(np.zeros((h, w), dtype=np.float32))
                self.data.append(np.array(segmentations))
                self.gts.append(gt_image)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        c, h, w = self.data[index].shape
        data = np.reshape(self.data[index], (1, h, w, c))
        augmented = self.transform(images=data, mask=self.gts[index])
        return augmented["images"], augmented["mask"].unsqueeze(-3).unsqueeze(-3)


class EnsembleDatasetC1(Dataset):
    """
    Ensemble dataset data structure C1.

    Input: crop image with the normalized overlap of competitors segmentations.
    Label: crop image of ground truth.
    """

    def __init__(
        self,
        ensemble_parquet_path,
        split,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        # load dataframe
        df = ext.load_parquet(ensemble_parquet_path)

        self.version = Version.C1
        if transform is None:
            self.transform = A.Compose([A.ToTensorV2()])
        else:
            self.transform = transform
        self.data = []
        self.gts = []

        # fill tensors with actual data
        for index, row in enumerate(df.itertuples()):
            if split != "all":
                if row.split != split:
                    continue
            # load the image
            composed_image = tifffile.imread(row.image_path)  # type: ignore
            # split the composed image

            # albumentations require (H, W, C) for images
            segmentation = (
                composed_image[0].astype(dtype=np.float32) / 255
            )  # scale down to [0,1]
            gt_image = (
                composed_image[1].astype(dtype=np.float32) / 255
            )  # scale down to [0,1]
            self.data.append(segmentation)
            self.gts.append(gt_image)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # return self.transform(self.data[index]), self.transform(self.gts[index])
        augmented = self.transform(image=self.data[index], mask=self.gts[index])
        return augmented["image"], augmented["mask"].unsqueeze(-3)


class EnsembleDatasetC2(Dataset):
    """
    Ensemble dataset data structure C2.

    Input:
        - crop image with the normalized overlap of competitors segmentations.
        - crop image of the cell.
    Label: crop image of ground truth.
    """

    def __init__(
        self,
        ensemble_parquet_path,
        split,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        # load dataframe
        df = ext.load_parquet(ensemble_parquet_path)

        self.version = Version.C2
        if transform is None:
            self.transform = A.Compose([A.ToTensorV2()])
        else:
            self.transform = transform
        self.data = []
        self.gts = []

        # fill tensors with actual data
        for index, row in enumerate(df.itertuples()):
            if split != "all":
                if row.split != split:
                    continue
            # load the image
            composed_image = tifffile.imread(row.image_path)  # type: ignore
            # split the composed image

            # albumentations require (H, W, C) for images
            segmentation = (
                composed_image[0].astype(dtype=np.float32) / 255
            )  # scale down to [0,1]
            gt_image = (
                composed_image[1].astype(dtype=np.float32) / 255
            )  # scale down to [0,1]
            raw_image = (
                composed_image[2].astype(dtype=np.float32) / 255
            )  # scale down to [0,1]
            self.data.append(np.array([segmentation, raw_image]))
            self.gts.append(gt_image)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        c, h, w = self.data[index].shape
        data = np.reshape(self.data[index], (1, h, w, c))
        augmented = self.transform(images=data, mask=self.gts[index])
        return augmented["images"], augmented["mask"].unsqueeze(-3).unsqueeze(-3)

        # augmented = self.transform(image=self.data[index], mask=self.gts[index])
        # return augmented["image"], augmented["mask"].unsqueeze(-3)


def benchmark_EnsembleDataset(path, epochs=1000):
    en_dataset = EnsembleDatasetC1(path, "train")
    print(en_dataset)

    start = time.time()
    for __ in tqdm(range(epochs), total=epochs):
        for index in range(en_dataset.__len__()):
            en_dataset.__getitem__(index)
    end = time.time()

    print(f"{(end-start)}s")
