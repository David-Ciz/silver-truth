import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
import torch
from torch import optim
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from src.silver_truth.ensemble.act_functions import LevelTrigger
from src.silver_truth.ensemble.models_loss_type import LossType
from enum import Enum

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


class ModelType(Enum):
    Unet = 1
    UnetPlusPlus = 2
    FPN = 3
    PSPNet = 4
    DeepLabV3 = 5
    DeepLabV3Plus = 6
    LinkNet = 7
    MAnet = 8
    PAN = 9
    UPerNet = 10
    Segformer = 11
    DPT = 12
    Unet_Mult_Input = 13
    Unet_Dynamic = 14


class SMP_Model(pl.LightningModule):
    def __init__(self, model_type: ModelType, device: torch.device, num_inputs: int=1):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._get_model(model_type, num_inputs)
        self.level_trigger = LevelTrigger(device)
        self.loss_type = LossType.MSE
        # self.loss_function = DiceLoss("binary", from_logits=True)

    def _get_model(self, model_type: ModelType, num_inputs: int):
        # Bug on load_from_checkpoint() --> model_type != ModelType.[type]
        match ModelType(model_type.value):
            case ModelType.Unet:
                return smp.Unet(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.UnetPlusPlus:
                return smp.UnetPlusPlus(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.FPN:
                return smp.FPN(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.PSPNet:
                return smp.PSPNet(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.DeepLabV3:
                return smp.DeepLabV3(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.DeepLabV3Plus:
                return smp.DeepLabV3Plus(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.LinkNet:
                return smp.Linknet(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.MAnet:
                return smp.MAnet(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.PAN:
                return smp.PAN(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.UPerNet:
                return smp.UPerNet(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.Segformer:
                return smp.Segformer(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.DPT:
                return smp.DPT(
                    encoder_name="resnet34", encoder_weights=None, in_channels=num_inputs
                )
            case ModelType.Unet_Mult_Input:
                raise Exception("Error: Network not implemente here.")
            case ModelType.Unet_Dynamic:
                raise Exception("Error: Network not implemente here.")
    

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
