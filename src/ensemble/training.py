# Register the models
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.dataset import Subset
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping
import mlflow
import matplotlib.pyplot as plt
from src.ensemble.datasets import EnsembleDatasetV1
from src.ensemble.models_loss_type import LossType
from ensemble.model_ae_v1 import Autoencoder_v1


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
    with torch.no_grad():
        model.eval()
        reconst_imgs = model(input_set)
        loss = F.mse_loss(reconst_imgs, target_set, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        model.train()
        return loss.item()


class EvaluationCallback(Callback):
    def __init__(self, train_set, val_set, every_n_epochs=1):
        super().__init__()
        self.train_inputs, self.train_targets = train_set[0], train_set[1]
        self.val_inputs, self.val_targets = val_set[0], val_set[1]
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            train_result = _evaluate_model(pl_module, self.train_inputs, self.train_targets)
            val_result = _evaluate_model(pl_module, self.val_inputs, self.val_targets)
            # Log the loss metric
            mlflow.log_metric("train_loss", value=train_result, step=trainer.current_epoch+1)
            mlflow.log_metric("val_loss", value=val_result, step=trainer.current_epoch+1)
            print(f"val_loss: {val_result}")


def _get_eval_sets(subset: Subset):
    return  torch.stack([subset.dataset[i][0] for i in subset.indices], dim=0), \
            torch.stack([subset.dataset[i][1] for i in subset.indices], dim=0)


def _get_stacked_images(dataset, num):
    return  torch.stack([dataset[i][0] for i in range(num)], dim=0), \
            torch.stack([dataset[i][1] for i in range(num)], dim=0)


def _train_model(
        max_epochs, 
        checkpoint_path, 
        train_dataset, 
        val_dataset, 
        train_loader, 
        val_loader, 
        test_loader, 
        latent_dim,
        model_input_size,
    ):
    loss_type = LossType.MSE
    model = Autoencoder_v1(num_inputs=1, num_channels=64, latent_dim=latent_dim, input_size=model_input_size, loss_type=loss_type)
    mlflow.log_param("model", model)
    mlflow.log_param("loss_type", loss_type)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, f"modelV1_dsV1.00_{latent_dim}"),
        deterministic=True,
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            #ModelCheckpoint(save_weights_only=True),
            #GenerateCallback(_get_stacked_images(train_dataset, 8), every_n_epochs=10),
            #LearningRateMonitor("epoch"),
            EvaluationCallback(
                _get_eval_sets(train_dataset),
                _get_eval_sets(val_dataset),
            ),
            EarlyStopping(monitor="val_loss", patience=10)
        ],
    )

    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"val": val_result, "test": test_result}
    return model, result



def _visualize_reconstructions(model, train_set):
    train_imgs, gt_images = train_set[0], train_set[1]
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(train_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([gt_images, reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(14, 10))
    plt.title(f"Reconstructed from {model.hparams.latent_dim} latents")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def run() -> None:
    rand_seed = 42
    pl.seed_everything(seed=rand_seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    """"""

    # set the input size for the model
    model_input_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(), 
        #transforms.Normalize((0.5,), (0.5,)), 
        transforms.Resize((model_input_size, model_input_size)),
    ])
    mlflow.log_param("dataset_input_transform", transform)
    mlflow.log_param("dataset_target_transform", transform)

    # get dataset
    #dataset_path = os.path.join(os.getcwd(), "data/ensemble_data/datasets/v1.00")
    dataset_path = "data/ensemble_data/datasets/v1.00"
    parquet_filename = "ensemble_dataset_v1.00.parquet"
    parquet_path = os.path.join(dataset_path, parquet_filename)
    checkpoint_path = os.path.join(dataset_path, "training_logs")
    ensemble_dataset = EnsembleDatasetV1(parquet_path, transform, transform)
    # split dataset
    dataset_split = [0.7, 0.15, 0.15]
    train_set, val_set, test_set = torch.utils.data.random_split(ensemble_dataset, dataset_split)
    # dataloaders
    batch_size = 20
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    max_epochs = 100
    latent_dim = 96

    # log parameters
    params = {
        "rand_seed": rand_seed,
        "dataset_split": dataset_split,
        "latent_dim": latent_dim,
        "parquet_path": parquet_path,
        "ensemble_dataset": ensemble_dataset,
        "batch_size": batch_size,
        "model_input_size": model_input_size,
    }
    mlflow.log_params(params)


    # train model
    model, result = _train_model(
        max_epochs,
        checkpoint_path, 
        train_set,
        val_set,
        train_loader, 
        val_loader,
        test_loader,
        latent_dim,
        model_input_size,
    )

    _visualize_reconstructions(model, _get_stacked_images(val_set, 8))