#import torch
#import lightning as pl
import logging
from ensemble.dataset_builds import build_dataset, Version
import src.ensemble.external as ext


# Basic Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#print("Device:", device)

#seed_value = 42
#pl.seed_everything(seed_value)



def build_required_datasets(
        ensemble_dataset_version=Version.V1
):
    original_dataset = "BF-C2DL-HSC"
    dataset_dataframe_path = f"data/dataframes/{original_dataset}_dataset_dataframe.parquet"
    qa_output_path = f"data/ensemble_data/qa/qa_images_{original_dataset}"
    qa_parquet_path = f"data/ensemble_data/qa/qa_{original_dataset}.parquet"

    ext.build_qa_dataset(
    dataset_dataframe_path, 
    qa_output_path, 
    qa_parquet_path)

    ext.compress_images(qa_output_path)

    ensemble_datasets_path = "data/ensemble_data/datasets"
    build_dataset(qa_parquet_path, ensemble_datasets_path, ensemble_dataset_version)


def run_ensemble_experiment(model: str, parameters: dict):
    pass
    


