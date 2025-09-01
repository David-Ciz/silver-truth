import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import os

# Config
QA_PARQUET = "qa_BF-C2DL-MuSC_dataset.parquet"  # Change as needed
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 240  # EfficientNet-B1 default input size

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
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels.iloc[idx]
        return image, label

class EfficientNetB1Lightning(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')
        self.model._fc = torch.nn.Linear(self.model._fc.in_features, 1)  # Regression output
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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

    model = EfficientNetB1Lightning(learning_rate=LEARNING_RATE)

    class PrintLossCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            train_loss = metrics.get('train_loss')
            val_loss = metrics.get('val_loss')
            epoch = trainer.current_epoch
            msg = f"Epoch {epoch+1}: "
            if train_loss is not None:
                msg += f"train_loss={train_loss:.4f} "
            if val_loss is not None:
                msg += f"val_loss={val_loss:.4f} "
            print(msg)

    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator='auto', devices='auto', log_every_n_steps=1, callbacks=[PrintLossCallback()])
    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(model, test_loader)
    if test_results and 'test_loss' in test_results[0]:
        print(f"Final test_loss: {test_results[0]['test_loss']:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()
