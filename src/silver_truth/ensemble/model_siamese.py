import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
import torch
from torch import optim
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from silver_truth.ensemble.act_functions import LevelTrigger
from silver_truth.ensemble.models_loss_type import LossType


class SiameseUnet(pl.LightningModule):
    def __init__(self, device: torch.device, max_competitors: int = 32):
        super().__init__()
        # We still use SMP, but we will call encoder and decoder separately
        # in_channels=1 because we process each competitor mask INDEPENDENTLY
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )
        self.level_trigger = LevelTrigger(device)
        self.loss_type = LossType.MSE
        self.max_competitors = max_competitors

    def forward(self, x):
        """
        Input x shape: (Batch, N_Competitors, H, W)
        We want to process each Competitor independently through the same Encoder.
        """
        b, n, h, w = x.shape
        # 1. BATCH FOLDING: Reshape to treat Competitors as part of the Batch
        # New shape: (Batch * N, 1, H, W)
        x_reshaped = x.view(b * n, 1, h, w)
        # 2. ENCODER PASS (Weight Shared)
        # SMP encoders return a list of features (for skip connections)
        # features[0] is the high-res input, features[-1] is the bottleneck
        features_list = self.model.encoder(x_reshaped)
        # 3. FEATURE FUSION (Max Pooling)
        # We need to reshape back and fuse EACH feature map in the list (for skip connections)
        fused_features = []
        for feat in features_list:
            # feat shape: (B*N, C, H_feat, W_feat)
            _, c, h_f, w_f = feat.shape
            # Unfold: (B, N, C, H_feat, W_feat)
            feat_unfolded = feat.view(b, n, c, h_f, w_f)
            # Max Pool over the 'N' dimension (Competitors)
            # This selects the strongest feature across all competitors for this spatial location.
            # It naturally ignores Zero-Padded inputs (as long as features are ReLU'd/positive)
            fused_feat, _ = torch.max(feat_unfolded, dim=1)
            fused_features.append(fused_feat)
        # 4. DECODER PASS (Runs once on the fused representation)
        decoder_output = self.model.decoder(*fused_features)
        # 5. SEGMENTATION HEAD
        masks = self.model.segmentation_head(decoder_output)
        if not self.training:
            masks = self.level_trigger(masks)
        return masks

    def _get_reconstruction_loss(self, batch):
        x, y = batch  # x is (B, N, H, W), y is (B, 1, H, W)
        x_hat = self(x)
        return self.get_loss(x_hat, y)

    def get_loss(self, x, y):
        # Your existing loss logic
        loss = F.mse_loss(x, y, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return OptimizerLRSchedulerConfig(
            {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        )

    # ... training/val/test steps remain the same ...
