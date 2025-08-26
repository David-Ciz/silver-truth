import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pytorch_lightning as pl
import os

# Config
QA_PARQUET = "qa_BF-C2DL-MuSC_dataset.parquet"  # Change as needed
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.001
IMG_SIZE = 224  # ResNet default input size

class QADataset(Dataset):
    def __init__(self, parquet_path, transform=None):
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform
        self.img_paths = self.df['stacked_path'] if 'stacked_path' in self.df.columns else self.df['image_path']
        self.labels = self.df['label'] if 'label' in self.df.columns else np.zeros(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.img_paths.iloc[idx]
        image = np.array(Image.open(img_path))
        # If the image is 2D, make it two-channel (duplicate)
        if image.ndim == 2:
            image = np.stack([image, image], axis=-1)
        # If it has more than 2 channels, take only the first two
        elif image.ndim == 3 and image.shape[2] > 2:
            image = image[:, :, :2]
        # Convert back to PIL for transforms
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        # Output must be [channels, height, width]
        if isinstance(image, torch.Tensor) and image.shape[0] != 2:
            image = image.permute(1, 2, 0).permute(2, 0, 1)
        label = self.labels.iloc[idx]
        return image, label

class ResNetLightning(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet50(pretrained=True)
        # Replace first conv layer to accept 2 channels
        self.model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)  # Regression output
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def main():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # No normalization, as input is 2 channels and not ImageNet RGB
    ])
    train_path = "qa_output_split/train.parquet"
    val_path = "qa_output_split/val.parquet"
    test_path = "qa_output_split/test.parquet"

    train_dataset = QADataset(train_path, transform=transform)
    val_dataset = QADataset(val_path, transform=transform)
    test_dataset = QADataset(test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = ResNetLightning(learning_rate=LEARNING_RATE)

    # Store losses for visualization
    train_losses = []
    val_losses = []
    # Remove per-epoch test loss computation (not allowed during training)
    class PrintLossCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            train_loss = metrics.get('train_loss')
            val_loss = metrics.get('val_loss')
            epoch = trainer.current_epoch
            msg = f"Epoch {epoch+1}: "
            if train_loss is not None:
                msg += f"train_loss={train_loss:.4f} "
                train_losses.append(train_loss.cpu().item())
            if val_loss is not None:
                msg += f"val_loss={val_loss:.4f} "
                val_losses.append(val_loss.cpu().item())
            print(msg)

    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator='auto', devices='auto', log_every_n_steps=1, callbacks=[PrintLossCallback()])
    trainer.fit(model, train_loader, val_loader)

    # Final test loss
    test_results = trainer.test(model, test_loader)
    final_test_loss = None
    if test_results and 'test_loss' in test_results[0]:
        final_test_loss = test_results[0]['test_loss']
        print(f"Final test_loss: {final_test_loss:.4f}")

    print("Training finished.")

    # Print loss visualization in terminal
    print("\nLosses per epoch:")
    print("Epoch | Train Loss | Val Loss")
    for i in range(len(train_losses)):
        t = train_losses[i] if i < len(train_losses) else None
        v = val_losses[i] if i < len(val_losses) else None
        print(f"{i+1:5d} | {t:.4f}     | {v:.4f}")
    if final_test_loss is not None:
        print(f"\nFinal test_loss: {final_test_loss:.4f}")

if __name__ == "__main__":
    main()
