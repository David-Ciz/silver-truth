import logging
import mlflow
from ensemble.databanks_builds import build_databank, Version
import ensemble.envs as envs
import src.ensemble.external as ext
import src.ensemble.training as training

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format="\n%(asctime)s ### %(levelname)s --> %(message)s", force=True)
_logger = logging.getLogger(__name__)


def build_ensemble_databanks(datasets: list[str], from_qa: bool):
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
    build_databank(original_dataset, qa_parquet_path, ensemble_datasets_path)


def build_databanks(datasets: list[str]):
    # build origin databanks
        # qa
        # non-qa (copy most code from qa)
    # build ensemble databanks
        # qa
        # non-qa
    pass



def _set_mlflow_experiment(name: str) -> None:
    mlflow.set_tracking_uri(envs.mlflow_mlruns_path) # needs to be set before mlflow.get_experiment_by_name()
    # find mlflow experiment
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name, envs.mlflow_mlruns_path)


def run_experiment(name: str, parquet_file: str, parameters: dict):
    "Entry point for new Ensemble experiment."

    _set_mlflow_experiment(name)

    #TODO: prepare here database loading because lots of models use the same
    #      add parameter to pass train and val datasets

    try:
        # start mlflow run
        with mlflow.start_run() as mlflow_run:
            run_id = mlflow_run.info.run_id
            _logger.info(f"MLflow experiment \"{name}\": a run started with ID \"{run_id}\".")
            training.run(parquet_file)
    except Exception as ex:
            print(f"Error during Ensemble experiment: {ex}")
            mlflow.set_tag("status", "failed")












def build_required_datasets_DEPRECATED(ensemble_dataset_version=Version.C1):
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
    #build_dataset(qa_parquet_path, ensemble_datasets_path, ensemble_dataset_version)



