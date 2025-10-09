# Register the models
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
import matplotlib.pyplot as plt
from src.ensemble.datasets import EnsembleDatasetV1

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


"""
def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()
"""


class PrintValidationLossCallback(pl.Callback):
    def __init__(self, train_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = train_imgs[0]
        self.targets = train_imgs[1]
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(self.input_imgs)
                loss = F.mse_loss(reconst_imgs, self.targets, reduction="none")
                loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
                print(f"train_loss: {loss}")
                pl_module.train()
            

def get_train_images(train_dataset, num):
    return  torch.stack([train_dataset[i][0] for i in range(num)], dim=0), \
            torch.stack([train_dataset[i][1] for i in range(num)], dim=0)


def _train_model(max_epochs, checkpoint_path, train_dataset, train_loader, val_loader, test_loader, latent_dim):
    model = Autoencoder_v1(num_inputs=1, num_channels=64, width=32, height=32, latent_dim=latent_dim)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, f"modelV1_dsV1.00_{latent_dim}"),
        deterministic=True,
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            #ModelCheckpoint(save_weights_only=True),
            #GenerateCallback(get_train_images(train_dataset, 8), every_n_epochs=10),
            PrintValidationLossCallback(get_train_images(train_dataset, 10), every_n_epochs=1),
            #LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # type: ignore # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # type: ignore # Optional logging argument that we don't need


    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"val": val_result, "test": test_result}
    return model, result



def _visualize_reconstructions(model, train_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(train_imgs[0].to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([train_imgs[1], reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7, 4.5))
    plt.title(f"Reconstructed from {model.hparams.latent_dim} latents")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def run():
    pl.seed_everything(seed=42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    """"""

    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize((32,32))])

    # get dataset
    #dataset_path = os.path.join(os.getcwd(), "data/ensemble_data/datasets/v1.00")
    dataset_path = "data/ensemble_data/datasets/v1.00"
    parquet_filename = "ensemble_dataset_v1.00.parquet"
    checkpoint_path = os.path.join(dataset_path, "training_logs")
    train_dataset = EnsembleDatasetV1(os.path.join(dataset_path, parquet_filename), transform, transform)
    # split dataset
    train_set, val_set, test_set = torch.utils.data.random_split(train_dataset, [0.7, 0.15, 0.15])
    # dataloaders
    train_loader = data.DataLoader(train_set, batch_size=20, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=20, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=20, shuffle=False, drop_last=False, num_workers=4)

    max_epochs = 10
    model_dict = {}
    for latent_dim in [96]:#, 128, 256]:
        model_ld, result_ld = _train_model(
            max_epochs,
            checkpoint_path, 
            train_dataset, 
            train_loader, 
            val_loader,
            test_loader,
            latent_dim
        )
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
        print(f"latent_dim: {latent_dim},   {result_ld}")


    latent_dims = sorted(k for k in model_dict)
    val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]

    """

    fig = plt.figure(figsize=(6, 4))
    plt.plot(
        latent_dims, val_scores, "--", color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16
    )
    plt.xscale("log")
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title("Reconstruction error over latent dimensionality", fontsize=14)
    plt.xlabel("Latent dimensionality")
    plt.ylabel("Reconstruction error")
    plt.minorticks_off()
    plt.ylim(0, 100)
    plt.show()
    plt.waitforbuttonpress(0)

    """

    input_imgs = get_train_images(train_dataset, 4)
    for latent_dim in model_dict:
        _visualize_reconstructions(model_dict[latent_dim]["model"], input_imgs)