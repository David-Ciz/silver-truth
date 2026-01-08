from typing import Callable, Optional
import tifffile
from tqdm import tqdm
from enum import Enum
import torch
from torch.utils.data import Dataset
import src.ensemble.external as ext
import albumentations as A
import numpy as np
#from PIL import Image



class Version(Enum):
    A1 = 1  # gt, gt            [1,1]
    A2 = 2  # gt&raw, gt        [2,1]
    B1 = 3  # seg, gt           [1,1]
    B2 = 4  # seg&raw, gt       [2,1]
    B3 = 5  # seg+gt, gt        [1,1]
    C1 = 6  # norm_seg, gt      [1,1]
    C2 = 7  # norm_seg&raw, gt  [2,1]
    D1 = 8  # raw, gt           [1,1]



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
            composed_image = tifffile.imread(row.image_path) # type: ignore
            # split the composed image
            
            # albumentations require (H, W, C) for images
            segmentation = composed_image[0].astype(dtype=np.float32) / 255 # scale down to [0,1]
            gt_image = composed_image[1].astype(dtype=np.float32) / 255 # scale down to [0,1]
            self.data.append(segmentation)
            self.gts.append(gt_image)
            """
            # torchvision
            segmentation, gt_image = composed_image[0], composed_image[1]
            self.data.append(Image.fromarray(segmentation))
            self.gts.append(Image.fromarray(gt_image))
            """

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        #return self.transform(self.data[index]), self.transform(self.gts[index])
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
    def __init__(self, ensemble_parquet_path) -> None:
        super().__init__()
        # load dataframe
        df = ext.load_parquet(ensemble_parquet_path)

        self.version = Version.C2
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
    en_dataset = EnsembleDatasetC1(path, "train")
    print(en_dataset)

    start = time.time()
    for __ in tqdm(range(epochs), total=epochs):
        for index in range(en_dataset.__len__()):
            en_dataset.__getitem__(index)
    end = time.time()

    print(f"{(end-start)}s")