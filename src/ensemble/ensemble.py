import logging
import mlflow
from src.ensemble.datasets import EnsembleDatasetC1
from src.ensemble.databanks_builds import Version
import src.ensemble.databanks_builds as db_builds
import src.ensemble.envs as envs
import src.ensemble.external as ext
import src.ensemble.training as training
from src.ensemble.models import SMP_Model
import src.qa.preprocessing as  qa_pp
import src.ensemble.utils as utils
from src.data_processing.utils.parquet_utils import same_splits
import segmentation_models_pytorch as smp
import torch
import pandas as pd
import os
from enum import Enum


# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format="\n%(asctime)s ### %(levelname)s --> %(message)s", force=True)
_logger = logging.getLogger(__name__)


def build_databank(build_opt: dict, qa_parquet_path: str) -> str:
    """
    Builds the Ensemble databank.
    """
    # build ensemble dataset
    ensemble_parquet_path = db_builds.build_databank(build_opt, qa_parquet_path, utils.DATABANKS_DIR)

    # confirm that the splits are the same
    same_splits_result = same_splits(qa_parquet_path, ensemble_parquet_path)
    print("Same splits: ",same_splits_result)
    assert(same_splits_result)

    return ensemble_parquet_path


def build_analysis_databanks(dataset_name: str, qa_parquet_path: str, mode: str) -> None:
    """
    Build databanks that allow data visualization.
    Requires previous call to build_databanks().
    Parameter <mode> can be 'all', 'crop' or 'full'.
    """
    if mode == "all" or mode == "crop":
        db_builds.build_analysis_databank(qa_parquet_path, f"data/ensemble_data/qa/qa_{dataset_name}_viz")
    if mode == "all" or mode == "full":
        db_builds.build_analysis_databank_full(qa_parquet_path, f"data/ensemble_data/qa/qa_{dataset_name}_vizfull")



"""
def build_databanks(datasets: list[str]):
    # build origin databanks
        # qa
        # non-qa (copy most code from qa)
    # build ensemble databanks
        # qa
        # non-qa
    pass
"""


def _set_mlflow_experiment(name: str) -> None:
    mlflow.set_tracking_uri(envs.mlflow_mlruns_path) # needs to be set before mlflow.get_experiment_by_name()
    # find mlflow experiment
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name, envs.mlflow_mlruns_path)
    else:
        mlflow.set_experiment(name)


def run_experiment(name: str, databank_name: str, parquet_file: str, run_sequence: list[dict]):
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
                training.run(databank_name, run_params, parquet_file)
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


def generate_evaluation(model_path: str, databank_path: str, split_type: str = "test") -> str:
    """
    Generate a parquet file with the evaluation of the given model checkpoint against the given set of a databank.
    If split_type is "all", it creates a copy of the parquet file with the metrics added.
    """
    
    # load model
    #TODO: what about if it's other models
    #torch.serialization.add_safe_globals([ModelType])
    model = SMP_Model.load_from_checkpoint(model_path, device=utils.get_device(), weights_only=False)

    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).split(".ckpt")
    dataset_name = os.path.basename(databank_path).split(".parquet")
    output_parquet_path = os.path.join(model_dir, f"{dataset_name}_{model_name}_set-{split_type}.parquet")

    # load dataset
    #TODO: what if it's other dataset?
    dataset = EnsembleDatasetC1(databank_path, split_type)
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
    df = ext.load_parquet(databank_path)
    if split_type != "all":
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
        output_df = pd.DataFrame(data_list)
        # save results        
        output_df.to_parquet(output_parquet_path)
        
    else: # add metric to a copy of the input parquet file
        df[model_name+"_iou"] = iou.numpy()
        df[model_name+"_f1"] = f1.numpy()
        df.to_parquet(output_parquet_path)

    return output_parquet_path
