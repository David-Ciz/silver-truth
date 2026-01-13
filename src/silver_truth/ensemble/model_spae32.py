# Sparse Autoencoder, based on: https://github.com/IParraMartin/Sparse-Autoencoder/blob/main/sae.py

import torch
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from collections.abc import Callable
from silver_truth.ensemble.models_loss_type import LossType
from torchvision import transforms
from silver_truth.ensemble.act_functions import LevelTrigger
from torchmetrics.classification import BinaryJaccardIndex


class Encoder32(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: Callable = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
        super().__init__()
        c_hid = base_channel_size
        self.transform = transforms.Compose([transforms.Resize((32, 32)),])
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            nn.Sigmoid(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        x = self.transform(x)
        return self.net(x)


class Decoder32(nn.Module):
    def __init__(self, base_channel_size: int, latent_dim: int, last_act_fn: Callable, act_fn: Callable = nn.GELU):
        """Decoder.

        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()
        self.transform = transforms.Compose([transforms.Resize((64, 64)),])
        num_outputs = 1 # the output is a single "grayscale" image
        c_hid = base_channel_size

        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn(),)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_outputs, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            last_act_fn(),  # Tanh() for MSE, Sigmoid() for BCE
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        x = self.transform(x)
        return x


class SparseAutoencoder32(pl.LightningModule):
    def __init__(
        self,
        num_inputs: int,
        num_channels: int,
        latent_dim: int,
        sparsity_lambda=1e-4,
        sparsity_target=0.05,
    ):
        super().__init__()

        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target

        self.loss_type = LossType.MSE_KL
        last_decoder_act_fn = nn.Tanh

        # Creating encoder and decoder
        self.encoder = Encoder32(num_inputs, num_channels, latent_dim)
        self.decoder = Decoder32(num_channels, latent_dim, last_decoder_act_fn)

        self.level_trigger = LevelTrigger()
        self.jaccard = BinaryJaccardIndex()

        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(3, num_inputs, 64, 64)
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        x = self.encoder(x)
        x = self.decoder(x)
        if not self.training:
            # filter out values below high_pass_filter threshold
            x = self.level_trigger(x)
        return x
    
    def forward_full(self, x):
        x_enc = self.encoder(x)
        x_hat = self.decoder(x_enc)
        if not self.training:
            # filter out values below high_pass_filter threshold
            x_hat = self.level_trigger(x_hat)
        return x_hat, x_enc

    def _get_reconstruction_loss(self, batch):
        x, y = batch
        x_hat, x_enc = self.forward_full(x)
        return self.get_loss(x_hat, y, x_enc)

    def sparsity_penalty(self, encoded):
        #rho_hat = torch.mean(encoded, dim=0)
        rho_hat = encoded.sum(dim=[0, 1]).mean(dim=[0])
        rho = self.sparsity_target
        epsilon = 1e-8
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)
        kl_divergence = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        sparsity_penalty = torch.sum(kl_divergence)
        return self.sparsity_lambda * sparsity_penalty

    """
    Create a custom loss that combine mean squared error (MSE) loss 
    for reconstruction with the sparsity penalty.
    """
    def get_loss(self, x, y, x_enc):
        loss = F.mse_loss(x, y, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        sparsity_loss = self.sparsity_penalty(x_enc)
        return loss + sparsity_loss
    
    #TODO: experiment with a more complex loss
    # -> Remove if it doesn't improve learning
    def get_loss_with_jaccard(self, x, y, x_enc):
        loss = self.get_loss(x, y, x_enc)
        # calculate jaccard
        iou = self.jaccard(self.level_trigger(x), y)
        iou_loss = loss * (1 - iou)
        return loss + iou_loss
 
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return OptimizerLRSchedulerConfig({"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"})

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
