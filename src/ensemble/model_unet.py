from typing import Any, Callable, Dict, Sequence
from segmentation_models_pytorch.losses import DiceLoss
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch import nn, optim
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from ensemble.act_functions import LevelTrigger
from ensemble.models_loss_type import LossType

"""
smp.Unet.loss_type = property(lambda self:self._loss_type,                              # type: ignore
                              lambda self, value: setattr(self, '_loss_type' ,value))
Unet_smp = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
)
Unet.loss_type = LossType.SMP   # type: ignore
"""

class Unet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1)
        self.level_trigger = LevelTrigger()
        self.loss_type = LossType.DICE
        self.loss_function = DiceLoss("binary", from_logits=True)
    
    def forward(self, x):
        x = self.model(x)
        if not self.training:
            # filter out values below high_pass_filter threshold
            x = self.level_trigger(x)
        return x
    
    def _get_reconstruction_loss(self, batch):
        x, y = batch
        x_hat = self.model(x)
        return self.get_loss(x_hat, y)
    
    def get_loss(self, x, y):
        #loss = F.mse_loss(x, y, reduction="none")
        #loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        loss = self.loss_function(x, y)
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
    