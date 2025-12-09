import logging
import mlflow
from src.ensemble.datasets import EnsembleDatasetC1
from src.ensemble.databanks_builds import build_databank, Version
import src.ensemble.envs as envs
import src.ensemble.external as ext
import src.ensemble.training as training
from src.ensemble.models import SMP_Model
import src.ensemble.utils as utils
import segmentation_models_pytorch as smp
import torch
import pandas as pd


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
    else:
        mlflow.set_experiment(name)


def run_experiment(name: str, parquet_file: str, run_sequence: list[dict]):
    "Entry point for new Ensemble experiment."

    _set_mlflow_experiment(name)

    #TODO: prepare here database loading because lots of models use the same
    #      add parameter to pass train and val datasets

    for run_params in run_sequence:
        try:
            # start mlflow run
            with mlflow.start_run() as mlflow_run:
                run_id = mlflow_run.info.run_id
                _logger.info(f"MLflow experiment \"{name}\": a run started with ID \"{run_id}\".")
                training.run(run_params, parquet_file)
        except Exception as ex:
                print(f"Error during Ensemble experiment: {ex}")
                mlflow.set_tag("status", "failed")


def find_best_ensemble(models_path, val_set):
    pass


#TODO: same as in training
def _get_eval_sets(dataset):
    imgs, gts = [],[]
    for i in range(len(dataset)):
        img, gt = dataset[i]
        imgs.append(img)
        gts.append(gt)
    return torch.stack(imgs, dim=0), torch.stack(gts, dim=0)


def generate_evaluation(model_path: str, dataset_path: str, split_type: str = "test") -> str:
    """
    Generate a parquet file with the evaluation of the given model checkpoint against the test set of the given dataset.
    """
    
    # load model
    #TODO: what about if it's other models
    model = SMP_Model.load_from_checkpoint(model_path, device=utils.get_device())

    # load dataset
    #TODO: what if it's other dataset?
    dataset = EnsembleDatasetC1(dataset_path, split_type)
    input_set, target_set = _get_eval_sets(dataset)
    input_set = input_set.to(model.device)
    target_set = target_set.to(model.device)

    with torch.no_grad():
        model.eval()
        # inference
        reconst_imgs = model(input_set)

    # calculate metrics
    tp, fp, fn, tn = smp.metrics.get_stats(reconst_imgs.long(), target_set.long(), mode='binary', threshold=0.5) #type: ignore
    iou = smp.metrics.iou_score(tp, fp, fn, tn)
    f1 = smp.metrics.f1_score(tp, fp, fn, tn)
    #iou_total = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    #f1_total = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

    # create dataframe
    data_list = []
    # load dataframe
    df = ext.load_parquet(dataset_path)
    df = df[df["split"]==split_type]
    
    for index, row in enumerate(df.itertuples()):
        data_list.append(
        {
            "image_path": row.image_path,
            "tp": tp[index].item(),
            "fp": fp[index].item(),
            "fn": fn[index].item(),
            "tn": tn[index].item(),
            "iou": iou[index].item(),
            "f1": f1[index].item(),
        })

    # save results
    #TODO: improve naming, add dataset name
    output_parquet_path = model_path[:-5] + f"_{split_type}" + ".parquet" # remove "ckpt" filetype and add "parquet"
    output_df = pd.DataFrame(data_list)
    output_df.to_parquet(output_parquet_path)
    return output_parquet_path








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



