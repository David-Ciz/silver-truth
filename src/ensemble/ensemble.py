import logging
import mlflow
from ensemble.dataset_builds import build_dataset, Version
import src.ensemble.external as ext
import src.ensemble.training as training

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format="\n%(asctime)s ### %(levelname)s --> %(message)s", force=True)
logger = logging.getLogger(__name__)


def build_required_datasets(ensemble_dataset_version=Version.V1):
    original_dataset = "BF-C2DL-HSC"
    dataset_dataframe_path = f"data/dataframes/{original_dataset}_dataset_dataframe.parquet"
    qa_output_path = f"data/ensemble_data/qa/qa_images_{original_dataset}"
    qa_parquet_path = f"data/ensemble_data/qa/qa_{original_dataset}.parquet"

    # build required qa dataset
    ext.build_qa_dataset(dataset_dataframe_path, qa_output_path, qa_parquet_path)
    # compress images to save space
    ext.compress_images(qa_output_path)
    # build ensemble dataset
    ensemble_datasets_path = "data/ensemble_data/datasets"
    build_dataset(qa_parquet_path, ensemble_datasets_path, ensemble_dataset_version)


def run_experiment(name: str, model: str, parameters: dict):
    "Entry point for new Ensemble experiment."

    # set mlflow experiment
    mlflow.set_experiment(name)

    # use automatic logging
    #mlflow.autolog()

    try:
        # start mlflow run
        with mlflow.start_run() as mlflow_run:
            run_id = mlflow_run.info.run_id
            logger.info(f"MLflow experiment \"{name}\": a run started with ID \"{run_id}\".")
            training.run()
    except Exception as ex:
            print(f"Error during Ensemble experiment: {ex}")
            mlflow.set_tag("status", "failed")

    


