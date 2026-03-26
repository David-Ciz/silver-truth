# Register the models
import os
import torch
import torch.utils.data as data
import torchvision
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from silver_truth.ensemble.model_unet_mult_input import Unet_Mult_Input
from silver_truth.ensemble.model_unet_dynamic import Unet_Dynamic
from silver_truth.ensemble.datasets import (
    Version,
    get_dataset_class,
    get_input_channels,
)
from silver_truth.ensemble.models_loss_type import LossType
from silver_truth.ensemble.models import ModelType, SMP_Model
import silver_truth.ensemble.utils as utils
import albumentations as A

# TODO: create config pipepline:
# config dictionary should be provided
_checkpoint_path = "data/ensemble_data/results/checkpoints"


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

""""""


def _evaluate_model_loader(model, dataloader):
    jaccard = BinaryJaccardIndex().to(model.device)
    f1_score = BinaryF1Score().to(model.device)
    total_loss = 0.0
    total_f1 = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        model.eval()
        for input_set, target_set in dataloader:
            input_set = input_set.to(model.device)
            target_set = target_set.to(model.device)
            if model.loss_type == LossType.BCE_KL:
                reconst_imgs, mean, logvar = model.forward_full(input_set)
                loss = model.get_loss(reconst_imgs, target_set, mean, logvar)
            elif model.loss_type == LossType.MSE_KL:
                reconst_imgs, x_enc = model.forward_full(input_set)
                loss = model.get_loss(reconst_imgs, target_set, x_enc)
            else:
                reconst_imgs = model(input_set)
                loss = model.get_loss(reconst_imgs, target_set)

            batch_size = int(input_set.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_iou += float(jaccard(reconst_imgs, target_set).item()) * batch_size
            total_f1 += float(f1_score(reconst_imgs, target_set).item()) * batch_size
            total_samples += batch_size

    model.train()
    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    return (
        total_loss / total_samples,
        total_f1 / total_samples,
        total_iou / total_samples,
    )


class EvaluationCallback(Callback):
    def __init__(self, val_loader, every_n_epochs=1):
        super().__init__()
        self.val_loader = val_loader
        self.every_n_epochs = every_n_epochs
        self.best_f1 = 0

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            val_loss, val_f1, val_iou = _evaluate_model_loader(
                pl_module, self.val_loader
            )
            if self.best_f1 < val_f1:
                self.best_f1 = val_f1
                mlflow.log_metric("best_val_f1", value=self.best_f1)
            mlflow.log_metric(
                "val_loss", value=val_loss, step=trainer.current_epoch + 1
            )
            mlflow.log_metric("val_f1", value=val_f1, step=trainer.current_epoch + 1)
            mlflow.log_metric("val_iou", value=val_iou, step=trainer.current_epoch + 1)
            print(f"val_loss: {val_loss}, val_f1: {val_f1}, val_iou: {val_iou}")


def _get_eval_sets(dataset, is_single_input):
    imgs, gts = [], []
    for i in range(len(dataset)):
        img, gt = dataset[i]
        imgs.append(img)
        gts.append(gt)
    return _batch_eval_tensors(imgs), _batch_eval_tensors(gts)


def _get_stacked_images(dataset, num, is_single_input):
    imgs, gts = [], []
    for i in range(min(num, len(dataset))):
        img, gt = dataset[i]
        imgs.append(img)
        gts.append(gt)
    return _batch_eval_tensors(imgs), _batch_eval_tensors(gts)


def _batch_eval_tensors(samples: list[torch.Tensor]) -> torch.Tensor:
    if not samples:
        raise ValueError("Cannot batch an empty sample list.")

    first = samples[0]
    # Multi-image datasets (for example B3) already carry a singleton batch axis.
    if first.ndim >= 4 and first.shape[0] == 1:
        return torch.cat(samples, dim=0)
    return torch.stack(samples, dim=0)


def _build_transform(augmentation: str, rand_seed: int) -> A.Compose:
    match augmentation:
        case "strong":
            transforms = [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ElasticTransform(alpha=30, sigma=5, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.ToTensorV2(),
            ]
        case "basic_vflip":
            transforms = [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ToTensorV2(),
            ]
        case "basic_brightness":
            transforms = [
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                A.ToTensorV2(),
            ]
        case "basic_noise":
            transforms = [
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
                A.ToTensorV2(),
            ]
        case "basic_vflip_brightness":
            transforms = [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                A.ToTensorV2(),
            ]
        case _:
            transforms = [
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.ToTensorV2(),
            ]

    return A.Compose(transforms, seed=rand_seed)


def _train_model(
    databank_name,
    run_params,
    train_dataset,
    val_dataset,
    train_loader,
    val_loader,
    test_loader,
    input_channels,
):
    device = utils.get_device()
    print("Device:", device)

    """
    model = Autoencoder32(
    #model = VariationalAutoencoder32(
    #model = SparseAutoencoder32(
        num_inputs=1, 
        num_channels=64, 
        latent_dim=latent_dim,
        loss_type=LossType.MSE,
    )
    """

    # TODO: following models not working
    # model_type = ModelType.PAN
    # model_type = ModelType.DPT

    model_type = run_params["model_type"]
    max_epochs = run_params["max_epochs"]
    encoder_name = run_params.get("encoder_name", "resnet34")
    encoder_weights = run_params.get("encoder_weights", None)
    init_from_checkpoint = run_params.get("init_from_checkpoint")
    if model_type == ModelType.Unet_Mult_Input:
        model_pl = Unet_Mult_Input(device)
    elif model_type == ModelType.Unet_Dynamic:
        model_pl = Unet_Dynamic(model_type, device)

    else:
        model_pl = SMP_Model(
            model_type,
            num_inputs=input_channels,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
        )

    if init_from_checkpoint:
        checkpoint = torch.load(init_from_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        load_result = model_pl.load_state_dict(state_dict, strict=True)
        print(f"Initialized model weights from checkpoint: {init_from_checkpoint}")
        print(f"Checkpoint load result: {load_result}")

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("model", model_pl.model)
    mlflow.log_param("loss_type", model_pl.loss_type)
    if init_from_checkpoint:
        mlflow.log_param("init_from_checkpoint", init_from_checkpoint)

    # Create a PyTorch Lightning trainer with the generation callback
    # Build safe filename for checkpoint to avoid nested quotes in f-string
    try:
        databank_suffix = databank_name.split("QA-")[1]
    except Exception:
        databank_suffix = databank_name

    checkpoint_filename = f"M{model_type.value}-{databank_suffix}"

    # Allow caller to override checkpoint directory; fall back to package default.
    ckpt_dirpath = run_params.get(
        "checkpoints_dir",
        f"{_checkpoint_path}/{databank_name}",
    )
    mlflow.log_param("checkpoints_dir", ckpt_dirpath)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(os.getcwd(), _checkpoint_path),
        deterministic=True,
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(
                filename=checkpoint_filename,
                dirpath=ckpt_dirpath,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_weights_only=True,
            ),
            # LearningRateMonitor("epoch"),
            EvaluationCallback(val_loader),
            EarlyStopping(monitor="val_loss", patience=10),
        ],
    )

    trainer.fit(model_pl, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model_pl, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model_pl, dataloaders=test_loader, verbose=False)
    result = {"val": val_result, "test": test_result}
    return model_pl, result


def _visualize_reconstructions(model, train_set):
    train_imgs, gt_images = train_set[0], train_set[1]
    model.to(utils.get_device())  # bypass LevelTrigger issue
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(train_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([gt_images, reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(
        imgs, nrow=8, normalize=True, value_range=(-1, 1)
    )
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(18, 13))
    plt.title("Reconstructions. Let's go!")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()
    plt.waitforbuttonpress(0)


def _visualize_dataset(subset):
    input_imgs, gt_images = subset[0], subset[1]

    # Plotting
    imgs = torch.stack([input_imgs, gt_images], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(
        imgs, nrow=8, normalize=True, value_range=(-1, 1)
    )
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(18, 13))
    plt.title("Check dataset.")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()
    plt.waitforbuttonpress(0)


def run(
    run_params: dict, parquet_file: Optional[str] = None, rand_seed: int = 42
) -> None:
    """
    Run a training session.
    With "remote", there's no visual feedback, such as image reconstructions.

    Parameters
    ----------
    run_params : dict
        Training parameters. Must contain ``"databank_opt"`` when ``parquet_file``
        is not provided (legacy path reconstruction behaviour).
    parquet_file : str, optional
        Explicit path to the databank parquet. When provided the legacy
        ``databank_opt``-based path reconstruction is skipped.
    rand_seed : int
        Random seed for reproducibility.
    """
    pl.seed_everything(seed=rand_seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if parquet_file is not None:
        parquet_path = parquet_file
        databank_opt = run_params.get("databank_opt", {})
        databank_name = Path(parquet_file).stem
    else:
        databank_opt = run_params["databank_opt"]
        databank_name = utils.get_databank_name(databank_opt)
        relative_parquet_path = f"data/ensemble_data/databanks/{databank_name}.parquet"
        parquet_path = f"{os.path.join(os.getcwd(), relative_parquet_path)}"

    latent_dim = None  # 32

    # ── Augmentation preset ──────────────────────────────────────────
    augmentation = run_params.get("augmentation", "basic")
    transform = _build_transform(augmentation, rand_seed)

    mlflow.log_param("dataset_transform", str(transform))
    mlflow.log_param("augmentation_preset", augmentation)

    # ── Dataset version ──────────────────────────────────────────────
    # CLI-specified version takes priority; fall back to databank_opt
    version_str = run_params.get("dataset_version", None)
    if version_str is not None:
        dataset_version = Version[version_str]
    else:
        dataset_version = databank_opt.get("dataset", Version.C1)
    input_channels = get_input_channels(dataset_version)

    # get datasets
    dataset_class = get_dataset_class(dataset_version)
    train_set = dataset_class(parquet_path, "train", transform)
    val_set = dataset_class(parquet_path, "validation")
    test_set = dataset_class(parquet_path, "test")

    # split dataset
    # dataset_split = [0.7, 0.15, 0.15]
    # train_set, val_set, test_set = torch.utils.data.random_split(ensemble_dataset, dataset_split)

    # TODO: note: use this to see the difference in learning with and without data augmentation
    # train_set.dataset = EnsembleDatasetC1(parquet_path, None)

    batch_size = int(run_params.get("batch_size", 7))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    # dataloaders
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,  # True
    )
    val_loader = data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # DEBUG only
    # _visualize_dataset(_get_eval_sets(val_set))

    # log parameters
    params = {
        "rand_seed": rand_seed,
        # "dataset_split": dataset_split,
        "latent_dim": latent_dim,
        "parquet_path": parquet_path,
        # "ensemble_dataset": ensemble_dataset,
        "batch_size": batch_size,
    }
    mlflow.log_params(params)

    # train model
    model, result = _train_model(
        databank_name,
        run_params,
        train_set,
        val_set,
        train_loader,
        val_loader,
        test_loader,
        input_channels,
    )

    print("Done.")
