# AE based: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
# VAE based: https://github.com/dariocazzani/pytorch-AE/blob/master/models/VAE.py

import torch
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from collections.abc import Callable
from src.ensemble.models_loss_type import LossType
from torchvision import transforms


class Encoder64(nn.Module):
    """Encoder.

        Args:
           num_inputs : Number of input images.
           num_channels : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
    def __init__(self, num_inputs: int, num_channels: int, latent_dim: int, act_fn: Callable = nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_inputs, num_channels, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
            act_fn(),
            nn.Conv2d(num_channels, 2 * num_channels, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * num_channels, 4 * num_channels, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(4 * num_channels, 4 * num_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(4 * num_channels, 4 * num_channels, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(start_dim=1),  # Image grid to single feature vector
            nn.Linear(4 * 16 * num_channels, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder64(nn.Module):
    def __init__(self, num_channels: int, latent_dim: int, last_act_fn: Callable, act_fn: Callable = nn.GELU):
        """Decoder.

        Args:
           num_channels : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(latent_dim, 4 * 16 * num_channels), act_fn(),)
        num_outputs = 1 # the output is a single "grayscale" image
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                4 * num_channels, 4 * num_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(4 * num_channels, 4 * num_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                4 * num_channels, 2 * num_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
               2 * num_channels, num_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            nn.ConvTranspose2d(
               num_channels, num_outputs, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32x32 => 64x64
            last_act_fn(), # Tanh() for MSE, Sigmoid() for BCE
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class Autoencoder64(pl.LightningModule):
    def __init__(
        self,
        num_inputs: int,
        num_channels: int,
        latent_dim: int,
        loss_type: LossType,
    ):
        super().__init__()

        if loss_type == LossType.MSE:
            last_decoder_act_fn = nn.Tanh
            self.loss_function = F.mse_loss
        elif loss_type == LossType.BCE:
            last_decoder_act_fn = nn.Sigmoid
            self.loss_function = F.binary_cross_entropy

        # Creating encoder and decoder
        self.encoder = Encoder64(num_inputs, num_channels, latent_dim)
        self.decoder = Decoder64(num_channels, latent_dim, last_decoder_act_fn)

        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(3, num_inputs, 64, 64)
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        x = self.encoder(x)
        return self.decoder(x)

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss."""
        x, y = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, y, reduction="sum")
        return loss
 
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
