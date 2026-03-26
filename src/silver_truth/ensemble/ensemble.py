import logging
from datetime import datetime
import mlflow
from silver_truth.ensemble.datasets import Version, get_dataset_class
import silver_truth.ensemble.databanks_builds as db_builds
import silver_truth.ensemble.envs as envs
import silver_truth.ensemble.external as ext
import silver_truth.ensemble.reconstruction as reconstruction
import silver_truth.ensemble.training as training
from silver_truth.ensemble.models import SMP_Model
import silver_truth.ensemble.utils as utils
from silver_truth.experiment_tracking import (
    ensure_mlflow_experiment,
    infer_dataset_name_from_text,
    infer_split_from_dataframe,
    resolve_mlflow_experiment_name,
    resolve_mlflow_tracking_uri,
    start_managed_mlflow_run,
    set_common_mlflow_tags,
)
from silver_truth.data_processing.utils.parquet_utils import same_splits
import segmentation_models_pytorch as smp
import torch
import torch.utils.data as data
import pandas as pd
import os
from typing import Dict, Optional, Union
from pathlib import Path


# Basic Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="\n%(asctime)s ### %(levelname)s --> %(message)s",
    force=True,
)
_logger = logging.getLogger(__name__)


def build_databank(
    build_opt: dict, qa_parquet_path: str, output_dir: Optional[str] = None
) -> str:
    """
    Builds the Ensemble databank.

    Parameters
    ----------
    output_dir : str, optional
        Override the default output directory (``utils.DATABANKS_DIR``).  Useful
        when building multiple databanks from different filtered parquets so they
        don't overwrite each other.
    """
    dest = output_dir if output_dir is not None else utils.DATABANKS_DIR
    # build ensemble dataset
    ensemble_parquet_path = db_builds.build_databank(build_opt, qa_parquet_path, dest)

    # confirm that the splits are the same (cell-level only; image-level has one row per image)
    if build_opt.get("aggregation_level", "cell") == "cell":
        same_splits_result = same_splits(qa_parquet_path, ensemble_parquet_path)
        print("Same splits: ", same_splits_result)
        assert same_splits_result

    return ensemble_parquet_path


def build_analysis_databanks(
    dataset_name: str, qa_parquet_path: str, mode: str
) -> None:
    """
    Build databanks that allow data visualization.
    Requires previous call to build_databanks().
    Parameter <mode> can be 'all', 'crop' or 'full'.
    """
    if mode == "all" or mode == "crop":
        db_builds.build_analysis_databank(
            qa_parquet_path, f"data/ensemble_data/qa/qa_{dataset_name}_viz"
        )
    if mode == "all" or mode == "full":
        db_builds.build_analysis_databank_full(
            qa_parquet_path, f"data/ensemble_data/qa/qa_{dataset_name}_vizfull"
        )


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
    tracking_uri = resolve_mlflow_tracking_uri(envs.mlflow_mlruns_path)
    experiment_name = resolve_mlflow_experiment_name(name) or name
    ensure_mlflow_experiment(
        experiment_name,
        tracking_uri=tracking_uri,
    )


def run_experiment(
    name: str,
    databank_name: str,
    parquet_file: str,
    run_sequence: list[dict],
    checkpoints_dir: Optional[str] = None,
):
    "Entry point for new Ensemble experiment."

    _set_mlflow_experiment(name)

    dataset_tag = infer_dataset_name_from_text([parquet_file, databank_name])
    split_tag = "unknown"
    try:
        split_df = pd.read_parquet(parquet_file, columns=["split"])
        split_tag = infer_split_from_dataframe(split_df)
    except Exception:
        pass

    parent_run_name = (
        f"{name}_{dataset_tag}_{split_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    with start_managed_mlflow_run(
        mlflow_tracking_uri=envs.mlflow_mlruns_path,
        mlflow_experiment=name,
        run_name=parent_run_name,
    ) as parent_run:
        set_common_mlflow_tags(dataset=dataset_tag, split=split_tag)
        mlflow.set_tag("run_kind", "experiment_parent")
        mlflow.set_tag("parent_scope", "dataset_split")
        mlflow.log_params(
            {
                "databank_name": databank_name,
                "parquet_file": parquet_file,
                "run_count": len(run_sequence),
            }
        )
        if checkpoints_dir is not None:
            mlflow.log_param("checkpoints_dir", checkpoints_dir)
        _logger.info(
            'MLflow experiment "%s": parent run started with ID "%s".',
            name,
            parent_run.info.run_id,
        )

        for run_index, run_params in enumerate(run_sequence, start=1):
            child_name = f"attempt_{run_index}"
            model_type = run_params.get("model_type")
            if model_type is not None:
                child_name = f"{child_name}_{model_type}"

            try:
                with start_managed_mlflow_run(
                    mlflow_tracking_uri=envs.mlflow_mlruns_path,
                    mlflow_experiment=name,
                    run_name=child_name,
                    nested=True,
                ) as mlflow_run:
                    set_common_mlflow_tags(dataset=dataset_tag, split=split_tag)
                    mlflow.set_tag("run_kind", "model_run")
                    run_id = mlflow_run.info.run_id
                    _logger.info(
                        'MLflow experiment "%s": child run started with ID "%s".',
                        name,
                        run_id,
                    )
                    # Merge checkpoints_dir into run_params so training.run can use it.
                    effective_params = dict(run_params)
                    if checkpoints_dir is not None:
                        effective_params["checkpoints_dir"] = checkpoints_dir
                    training.run(effective_params, parquet_file)
            except Exception as ex:
                print(f"Error during Ensemble experiment: {ex}")
                mlflow.set_tag("status", "failed")
                raise


def find_best_ensemble(models_path, val_set):
    pass


# TODO: same as in training
def _get_eval_sets(dataset):
    imgs, gts = [], []
    for i in range(len(dataset)):
        img, gt = dataset[i]
        imgs.append(img)
        gts.append(gt)
    return _batch_eval_tensors(imgs), _batch_eval_tensors(gts)


def _batch_eval_tensors(samples: list[torch.Tensor]) -> torch.Tensor:
    if not samples:
        raise ValueError("Cannot batch an empty sample list.")

    first = samples[0]
    if first.ndim >= 4 and first.shape[0] == 1:
        return torch.cat(samples, dim=0)
    return torch.stack(samples, dim=0)


def _resolve_dataset_version(
    model: SMP_Model, dataset_version: Optional[Union[str, Version]] = None
) -> Version:
    if isinstance(dataset_version, Version):
        return dataset_version
    if isinstance(dataset_version, str):
        return Version[dataset_version.upper()]

    num_inputs = int(getattr(model.hparams, "num_inputs", 1))
    if num_inputs == 3:
        return Version.C3
    if num_inputs == 2:
        return Version.C2
    return Version.C1


def generate_evaluation(
    model_path: str,
    databank_path: str,
    split_type: str = "test",
    output_dir: Optional[str] = None,
    dataset_version: Optional[Union[str, Version]] = None,
    batch_size: int = 8,
) -> str:
    """
    Generate a parquet file with the evaluation of the given model checkpoint against the given set of a databank.
    If split_type is "all", it creates a copy of the parquet file with the metrics added.

    Parameters
    ----------
    output_dir : str, optional
        Directory to write evaluation parquets into.  Defaults to the directory
        containing the checkpoint file.
    """

    # load model
    # TODO: what about if it's other models
    # torch.serialization.add_safe_globals([ModelType])
    model = SMP_Model.load_from_checkpoint(
        model_path, device=utils.get_device(), weights_only=False
    )

    model_dir = output_dir if output_dir is not None else os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    model_name = os.path.basename(model_path).split(".ckpt")[0]
    dataset_name = os.path.basename(databank_path).split(".parquet")[0]
    output_parquet_path = os.path.join(
        model_dir, f"{dataset_name}_{model_name}_set-{split_type}.parquet"
    )
    cell_level_output_parquet_path = os.path.join(
        model_dir, f"{dataset_name}_{model_name}_set-{split_type}_cell.parquet"
    )

    # load dataset
    # TODO: what if it's other dataset?
    resolved_dataset_version = _resolve_dataset_version(model, dataset_version)
    dataset_class = get_dataset_class(resolved_dataset_version)
    dataset = dataset_class(databank_path, split_type)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    reconst_parts = []
    tp_parts = []
    fp_parts = []
    fn_parts = []
    tn_parts = []

    with torch.no_grad():
        model.eval()
        for input_batch, target_batch in dataloader:
            input_batch = input_batch.to(model.device)
            target_batch = target_batch.to(model.device)
            reconst_batch = model(input_batch)
            tp, fp, fn, tn = smp.metrics.get_stats(
                reconst_batch, target_batch.long(), mode="binary", threshold=0.5
            )  # type: ignore
            reconst_parts.append(reconst_batch.detach().cpu())
            tp_parts.append(tp.detach().cpu())
            fp_parts.append(fp.detach().cpu())
            fn_parts.append(fn.detach().cpu())
            tn_parts.append(tn.detach().cpu())

    if not reconst_parts:
        raise ValueError(f"No samples found for split '{split_type}' in {databank_path}.")

    reconst_imgs = torch.cat(reconst_parts, dim=0)
    tp = torch.cat(tp_parts, dim=0)
    fp = torch.cat(fp_parts, dim=0)
    fn = torch.cat(fn_parts, dim=0)
    tn = torch.cat(tn_parts, dim=0)

    # calculate metrics
    iou = smp.metrics.iou_score(tp, fp, fn, tn)
    f1 = smp.metrics.f1_score(tp, fp, fn, tn)
    # iou_total = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    # f1_total = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

    # create dataframe
    data_list = []
    # load dataframe
    df = ext.load_parquet(databank_path)
    if split_type != "all":
        df = df[df["split"] == split_type]
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
                }
            )
        cell_level_df = pd.DataFrame(data_list)
        cell_level_df.to_parquet(cell_level_output_parquet_path)

        # For cell-level databanks with reconstruction metadata and labels,
        # evaluate after placing predicted crops back into full-image coordinates
        # using the same label-wise metric as the competitor baseline.
        if reconstruction.has_reconstruction_metadata(df) and "label" in df.columns:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reconstructed_dir = os.path.join(
                model_dir, f"{dataset_name}_{model_name}_set-{split_type}_reconstructed"
            )
            reconstructed_eval_df = (
                reconstruction.reconstruct_labeled_full_images_from_arrays(
                    databank_df=df.reset_index(drop=True),
                    predicted_crops=list(reconst_imgs),
                    output_dir=Path(reconstructed_dir),
                    threshold=0.5,
                )
            )
            if len(reconstructed_eval_df) > 0:
                reconstructed_eval_df.to_parquet(output_parquet_path)
            else:
                # Fallback to cell-level metrics if reconstruction produced no rows.
                cell_level_df.to_parquet(output_parquet_path)
        else:
            cell_level_df.to_parquet(output_parquet_path)

    else:  # add metric to a copy of the input parquet file
        df[model_name + "_iou"] = iou.detach().cpu().numpy()
        df[model_name + "_f1"] = f1.detach().cpu().numpy()
        df.to_parquet(output_parquet_path)

    return output_parquet_path


def evaluate_checkpoint(
    model_path: str,
    databank_path: str,
    split_type: str = "test",
    output_dir: Optional[str] = None,
    dataset_version: Optional[Union[str, Version]] = None,
    batch_size: int = 8,
) -> Dict[str, Union[float, int, str]]:
    """
    Run inference for a checkpoint on a databank split and return aggregated metrics.
    """
    output_parquet_path = generate_evaluation(
        model_path,
        databank_path,
        split_type,
        output_dir=output_dir,
        dataset_version=dataset_version,
        batch_size=batch_size,
    )
    output_df = pd.read_parquet(output_parquet_path)
    cell_output_parquet_path = output_parquet_path.replace(".parquet", "_cell.parquet")
    cell_output_df: Optional[pd.DataFrame] = None
    if os.path.exists(cell_output_parquet_path):
        try:
            cell_output_df = pd.read_parquet(cell_output_parquet_path)
        except Exception:
            cell_output_df = None

    if split_type == "all":
        model_name = os.path.basename(model_path).split(".ckpt")[0]
        iou_col = f"{model_name}_iou"
        f1_col = f"{model_name}_f1"
    else:
        iou_col = "iou"
        f1_col = "f1"

    evaluation_level = (
        "full_image_label"
        if {"reconstructed_path", "gt_image", "labels_scored"}.issubset(
            set(output_df.columns)
        )
        else "cell_crop"
    )

    if len(output_df) == 0:
        return {
            "output_parquet_path": output_parquet_path,
            "cell_output_parquet_path": cell_output_parquet_path,
            "split": split_type,
            "count": 0,
            "iou_mean": float("nan"),
            "f1_mean": float("nan"),
            "evaluation_level": evaluation_level,
        }

    summary: Dict[str, Union[float, int, str]] = {
        "output_parquet_path": output_parquet_path,
        "cell_output_parquet_path": cell_output_parquet_path,
        "split": split_type,
        "count": int(len(output_df)),
        "iou_mean": float(output_df[iou_col].mean()),
        "f1_mean": float(output_df[f1_col].mean()),
        "evaluation_level": evaluation_level,
    }
    if cell_output_df is not None and len(cell_output_df) > 0:
        summary["cell_count"] = int(len(cell_output_df))
        summary["cell_iou_mean"] = float(cell_output_df["iou"].mean())
        summary["cell_f1_mean"] = float(cell_output_df["f1"].mean())
    return summary
