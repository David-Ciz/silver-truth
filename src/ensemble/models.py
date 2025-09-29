# Register the models
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
#from ensemble.model_ae_v1 import Autoencoder
import tifffile

from src.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)

"""
def get_model(model: str, parameters: dict):
    if model == "ae-v1":
        #parameters.keys.
        #en_model = Autoencoder(num_inputs=1, num_channels=64, latent_dim=128)
        pass
    elif model == "ae-v2":

        pass
    else:
        raise Exception("Model name not found.")
"""

"""
pl.seed_everything(seed=42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
"""


class EnsembleDataset(Dataset):
    """
    Ensemble dataset data structure.
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



import time
from tqdm import tqdm

def benchmark_EnsembleDataset(path, epochs=10000):
    en_dataset = EnsembleDataset(path)
    print(en_dataset)

    start = time.time()
    for __ in tqdm(range(epochs), total=epochs):
        for index in range(en_dataset.__len__()):
            en_dataset.__getitem__(index)
    end = time.time()

    print(f"{(end-start)}s")


