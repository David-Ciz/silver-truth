import tifffile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from src.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)



class EnsembleDatasetV001(Dataset):
    """
    Ensemble dataset data structure V001.

    Input: crop image with the normalized overlap of competitors segmentations.
    Label: crop image of ground truth.
    """
    def __init__(self, ensemble_parquet_path) -> None:
        super().__init__()
        # load dataframe
        df = load_dataframe_from_parquet_with_metadata(ensemble_parquet_path)
        # get useful array properties
        img_count = len(df)
        img_size = df.iloc[0]["crop_size"]
 
        # create dataset tensors
        tensor_shape = (img_count, img_size, img_size)
        self.data = torch.empty(tensor_shape, dtype=torch.float32)
        self.gts = torch.empty(tensor_shape, dtype=torch.float32)
        
        # fill tensors with actual data
        for index, row in enumerate(df.itertuples()):
            # load the image
            composed_image = tifffile.imread(row.image_path) # type: ignore
            # split the composed image
            segmentation, gt_image = composed_image[0], composed_image[1]
            self.data[index, : , :] = torch.from_numpy(segmentation)
            self.gts[index, : , :] = torch.from_numpy(gt_image)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.gts[index])


class EnsembleDatasetV002(Dataset):
    """
    Ensemble dataset data structure V002.

    Input: 
        - crop image with the normalized overlap of competitors segmentations.
        - crop image of the cell.
    Label: crop image of ground truth.
    """
    def __init__(self, ensemble_parquet_path) -> None:
        super().__init__()
        # load dataframe
        df = load_dataframe_from_parquet_with_metadata(ensemble_parquet_path)
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
    en_dataset = EnsembleDatasetV001(path)
    print(en_dataset)

    start = time.time()
    for __ in tqdm(range(epochs), total=epochs):
        for index in range(en_dataset.__len__()):
            en_dataset.__getitem__(index)
    end = time.time()

    print(f"{(end-start)}s")