import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import os

 # Config
QA_PARQUET = "qa_BF-C2DL-MuSC_dataset.parquet"  # Change as needed
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001
IMG_SIZE = 64  # Assuming crop-size 64

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

class SimpleCNNLightning(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 64)
        self.fc2 = nn.Linear(64, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)

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
    ])
    # If you have split files, use them here
    train_path = "qa_output_split/train.parquet"
    val_path = "qa_output_split/val.parquet"
    test_path = "qa_output_split/test.parquet"

    train_dataset = QADataset(train_path, transform=transform)
    val_dataset = QADataset(val_path, transform=transform)
    test_dataset = QADataset(test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = SimpleCNNLightning(learning_rate=LEARNING_RATE)

    import matplotlib.pyplot as plt
    train_losses = []
    val_losses = []
    test_losses = []

    class PrintLossCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            train_loss = metrics.get('train_loss')
            epoch = trainer.current_epoch
            msg = f"Epoch {epoch+1}: "
            if train_loss is not None:
                msg += f"train_loss={train_loss:.4f} "
                train_losses.append(train_loss.cpu().item() if hasattr(train_loss, 'cpu') else float(train_loss))
            # Safe test loss calculation
            pl_module.eval()
            test_loss_epoch = 0.0
            n_batches = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(pl_module.device)
                    labels = labels.float().to(pl_module.device)
                    outputs = pl_module(images)
                    loss = pl_module.criterion(outputs, labels)
                    test_loss_epoch += loss.item()
                    n_batches += 1
            if n_batches > 0:
                test_loss_epoch /= n_batches
                msg += f" test_loss={test_loss_epoch:.4f}"
                test_losses.append(test_loss_epoch)
            pl_module.train()
            print(msg)
        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            val_loss = metrics.get('val_loss')
            epoch = trainer.current_epoch
            if val_loss is not None:
                val_losses.append(val_loss.cpu().item() if hasattr(val_loss, 'cpu') else float(val_loss))

    # Doporučení: před spuštěním smažte složku lightning_logs, aby se nenačítaly staré checkpointy
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=1,
        callbacks=[PrintLossCallback()],
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        default_root_dir=os.path.join(os.getcwd(), "lightning_logs_new")  # nový log adresář
    )
    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(model, test_loader)
    if test_results and 'test_loss' in test_results[0]:
        print(f"Final test_loss: {test_results[0]['test_loss']:.4f}")

    print("Training finished.")

    # Print model architecture
    print("\nModel architecture:")
    print(model)

    # Print main evaluation metric (MSE)
    if test_results and 'test_loss' in test_results[0]:
        print(f"Main evaluation metric (MSE): {test_results[0]['test_loss']:.4f}")

    # Plot losses for all epochs
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    if test_losses:
        plt.plot(test_losses, label='Test Loss (final, repeated)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Losses per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
