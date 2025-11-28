# Register the models
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.dataset import Subset
import torchvision
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
import mlflow
import matplotlib.pyplot as plt
from src.ensemble.datasets import EnsembleDatasetC1
from src.ensemble.models_loss_type import LossType
from ensemble.model_ae32 import Autoencoder32
from ensemble.model_ae64 import Autoencoder64
from ensemble.model_vae32 import VariationalAutoencoder32
from ensemble.model_spae32 import SparseAutoencoder32
from ensemble.model_unet import Unet
import albumentations as A
import segmentation_models_pytorch as smp

#TODO: create config pipepline: 
# config dictionary should be provided
#checkpoint_path = "data/ensemble_data/results/checkpoints222/"
_checkpoint_path = None

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

def _evaluate_model(model, input_set, target_set):
    jaccard = BinaryJaccardIndex().to(model.device)
    f1_score = BinaryF1Score().to(model.device)
    input_set = input_set.to(model.device)
    target_set = target_set.to(model.device)
    with torch.no_grad():
        model.eval()
        if model.loss_type == LossType.BCE_KL:
            reconst_imgs, mean, logvar = model.forward_full(input_set)
            # calculate model's loss
            loss = model.get_loss(reconst_imgs, target_set, mean, logvar)
        
        elif model.loss_type == LossType.MSE_KL:
            reconst_imgs, x_enc = model.forward_full(input_set)
            # calculate model's loss
            loss = model.get_loss(reconst_imgs, target_set, x_enc)
        else:
            # inference
            reconst_imgs = model(input_set)
            # calculate model's loss
            loss = model.get_loss(reconst_imgs, target_set)

        #tp, fp, fn, tn = smp.metrics.get_stats(reconst_imgs, target_set, mode='binary', threshold=0.5)
        #iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        #f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        # calculate IoU
        iou = jaccard(reconst_imgs, target_set)
        # calculate binary f1-score
        f1 = f1_score(reconst_imgs, target_set)
        model.train()
        return loss.item(), f1.item(), iou.item()


class EvaluationCallback(Callback):
    def __init__(self, train_set, val_set, every_n_epochs=1):
        super().__init__()
        self.train_inputs, self.train_targets = train_set[0], train_set[1]
        self.val_inputs, self.val_targets = val_set[0], val_set[1]
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            val_loss, val_f1, val_iou = _evaluate_model(pl_module, self.val_inputs, self.val_targets)
            mlflow.log_metric("val_loss", value=val_loss, step=trainer.current_epoch+1)
            mlflow.log_metric("val_f1", value=val_f1, step=trainer.current_epoch+1)
            mlflow.log_metric("val_iou", value=val_iou, step=trainer.current_epoch+1)
            print(f"val_loss: {val_loss}, val_f1: {val_f1}, val_iou: {val_iou}")


def _get_eval_sets(dataset):
    imgs, gts = [],[]
    for i in range(len(dataset)):
        img, gt = dataset[i]
        imgs.append(img)
        gts.append(gt)
    return torch.stack(imgs, dim=0), torch.stack(gts, dim=0)


def _get_stacked_images(dataset, num):
    imgs, gts = [],[]
    for i in range(num):
        img, gt = dataset[i]
        imgs.append(img)
        gts.append(gt)
    return torch.stack(imgs, dim=0), torch.stack(gts, dim=0)


def _train_model(
        max_epochs, 
        train_dataset, 
        val_dataset, 
        train_loader, 
        val_loader, 
        test_loader,
    ):
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
    model_pl = Unet(device)
    
    mlflow.log_param("model", model_pl.model)
    mlflow.log_param("loss_type", model_pl.loss_type)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(os.getcwd(), f'data/ensemble_data/results/checkpoints/model_{model_pl.loss_type.name}'),
        deterministic=True,
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=_checkpoint_path,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_weights_only=True),
            #LearningRateMonitor("epoch"),
            EvaluationCallback(
                _get_eval_sets(train_dataset),
                _get_eval_sets(val_dataset),
            ),
            EarlyStopping(monitor="val_loss",  patience=10)
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
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(train_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([gt_images, reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(18, 13))
    plt.title(f"Reconstructions. Let's go!")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()
    plt.waitforbuttonpress(0)


def _visualize_dataset(subset):
    input_imgs, gt_images = subset[0], subset[1]

    # Plotting
    imgs = torch.stack([input_imgs, gt_images], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(18, 13))
    plt.title(f"Check dataset.")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()
    plt.waitforbuttonpress(0)


def run(parquet_path: str, max_epochs: int=100, rand_seed: int=42, remote: bool=True) -> None:
    """
    Run a training session.
    With "remote", there's no visual feedback, such as image reconstructions.
    """
    pl.seed_everything(seed=rand_seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #print("Device:", device)
    """"""

    latent_dim = None#32

    transform = A.Compose([
        A.HorizontalFlip(),
        #A.Rotate(p=1.0),
        A.RandomRotate90(),
        A.ToTensorV2()
    ], seed=rand_seed)

    mlflow.log_param("dataset_transform", transform)

    # get dataset
    #ensemble_dataset = EnsembleDatasetC1(parquet_path, transform)
    train_set = EnsembleDatasetC1(parquet_path, "train", transform)
    val_set = EnsembleDatasetC1(parquet_path, "validation")
    test_set = EnsembleDatasetC1(parquet_path, "test")
    # split dataset
    #dataset_split = [0.7, 0.15, 0.15]
    #train_set, val_set, test_set = torch.utils.data.random_split(ensemble_dataset, dataset_split)

    #TODO: note: use this to see the difference in learning with and without data augmentation
    #train_set.dataset = EnsembleDatasetC1(parquet_path, None)
    
    # dataloaders
    batch_size = 20
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    if not remote and False:
        _visualize_dataset(_get_eval_sets(val_set))

    # log parameters
    params = {
        "rand_seed": rand_seed,
        #"dataset_split": dataset_split,
        "latent_dim": latent_dim,
        "parquet_path": parquet_path,
        #"ensemble_dataset": ensemble_dataset,
        "batch_size": batch_size,
    }
    mlflow.log_params(params)


    # train model
    model, result = _train_model(
        max_epochs,
        train_set,
        val_set,
        train_loader, 
        val_loader,
        test_loader,
    )

    if not remote:
        _visualize_reconstructions(model, _get_stacked_images(val_set, 16))
    print("Done.")