from typing import Callable, Optional
import tifffile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import src.ensemble.external as ext
from enum import Enum
from PIL import Image


class Version(Enum):
    V1 = 1
    V2 = 2


class EnsembleDatasetV1(Dataset):
    """
    Ensemble dataset data structure V1.

    Input: crop image with the normalized overlap of competitors segmentations.
    Label: crop image of ground truth.
    """
    def __init__(
            self, 
            ensemble_parquet_path, 
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__()
        # load dataframe
        df = ext.load_parquet(ensemble_parquet_path)
 
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.gts = []
        
        # fill tensors with actual data
        for index, row in enumerate(df.itertuples()):
            # load the image
            composed_image = tifffile.imread(row.image_path) # type: ignore
            # split the composed image
            segmentation, gt_image = composed_image[0], composed_image[1]
            self.data.append(Image.fromarray(segmentation))
            self.gts.append(Image.fromarray(gt_image))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img, gt = self.data[index], self.gts[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            gt = self.target_transform(gt)

        return img, gt


class EnsembleDatasetV2(Dataset):
    """
    Ensemble dataset data structure V2.

    Input: 
        - crop image with the normalized overlap of competitors segmentations.
        - crop image of the cell.
    Label: crop image of ground truth.
    """
    def __init__(self, ensemble_parquet_path) -> None:
        super().__init__()
        # load dataframe
        df = ext.load_parquet(ensemble_parquet_path)
        # get useful array properties
        img_count = len(df)
        img_size = df.iloc[0]["crop_size"]
 
        # create dataset tensors
        tensor_shape = (img_count, 2, img_size, img_size)
        self.data = torch.empty(tensor_shape, dtype=torch.float32)
        self.gts = torch.empty(tensor_shape, dtype=torch.float32)
        
        # fill tensors with actual data
        for index, row in enumerate(df.itertuples()):
            # load the image
            composed_image = tifffile.imread(row.image_path) # type: ignore
            # split the composed image
            segmentation, gt_image, cell = composed_image
            self.data[index, 0, : , :] = torch.from_numpy(segmentation)
            self.data[index, 1, : , :] = torch.from_numpy(cell)
            self.gts[index, : , :] = torch.from_numpy(gt_image)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.gts[index])



import time
from tqdm import tqdm

def benchmark_EnsembleDataset(path, epochs=1000):
    en_dataset = EnsembleDatasetV1(path)
    print(en_dataset)

    start = time.time()
    for __ in tqdm(range(epochs), total=epochs):
        for index in range(en_dataset.__len__()):
            en_dataset.__getitem__(index)
    end = time.time()

    print(f"{(end-start)}s")