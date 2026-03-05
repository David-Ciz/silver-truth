import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
import segmentation_models_pytorch as smp
import torch
from torch import optim
from torch.nn import functional as F
from silver_truth.ensemble.act_functions import LevelTrigger
from silver_truth.ensemble.models import ModelType
from silver_truth.ensemble.models_loss_type import LossType


class Unet_Dynamic(pl.LightningModule):
    def __init__(self, model_type: ModelType, device: torch.device):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name="resnet34", encoder_weights=None, in_channels=1
        )
        self.level_trigger = LevelTrigger(device)
        self.loss_type = LossType.MSE

    def forward(self, x):
        # x shape is expected as (B, N, H, W); current dynamic path consumes one sample at a time.
        res = []
        for x_n in x[0]:
            x_n = x_n.unsqueeze(0).unsqueeze(0)
            if res == []:
                res = self.model.encoder(x_n)
            else:
                features_list = self.model.encoder(x_n)
                for i, feat in enumerate(features_list):
                    res[i] += feat
        num_segs = len(x[0])
        for i, feat in enumerate(res):
            res[i] /= num_segs

        decoder_output = self.model.decoder(*res)
        x = self.model.segmentation_head(decoder_output)
        if not self.training:
            # filter out values below high_pass_filter threshold
            x = self.level_trigger(x)
        return x

    def _get_reconstruction_loss(self, batch):
        x, y = batch
        x_hat = self(x)
        return self.get_loss(x_hat, y)

    def get_loss(self, x, y):
        loss = F.mse_loss(x, y, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        # loss = self.loss_function(x, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return OptimizerLRSchedulerConfig(
            {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        )

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
