import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader import PETDataModule


class InceptionV3Model(pl.LightningModule):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()
        self.base_model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )
        self.base_model.aux_logits = False

        # Replace the top layer
        self.top_layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )
        self.base_model.Conv2d_1a_3x3 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1
        )
        self.base_model.fc = self.top_layer

    def forward(self, x):
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.666, patience=2, verbose=True, min_lr=1e-10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class ResNet50Model(pl.LightningModule):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Replace the top layer
        self.top_layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.base_model.fc = self.top_layer

    def forward(self, x):
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.666, patience=2, verbose=True, min_lr=1e-10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class DenseNet121Model(pl.LightningModule):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()
        self.base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # Replace the top layer
        self.top_layer = nn.Sequential(
            nn.Linear(1024, 512),  # DenseNet121 features 1024 output channels
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )
        self.base_model.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1
        )

        self.base_model.fc = self.top_layer

    def forward(self, x):
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.666, patience=2, verbose=True, min_lr=1e-10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def train_model(
    model_class, num_classes=5, batch_size=32, max_epochs=100, data_dir="./data"
):
    data_module = PETDataModule(data_dir, batch_size=batch_size)
    data_module.setup()

    model = model_class(num_classes=data_module.num_classes)

    logger = TensorBoardLogger(f"./lightning_logs/{model.__class__.__name__}/")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{model.__class__.__name__}/version_{logger.version}",
        filename="{model_name}-{epoch:02d}-{val_loss:.5f}-{val_acc:.5f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        verbose=True,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")

    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
        ],
        logger=logger,
        precision="16-mixed",
        enable_progress_bar=True,
        enable_checkpointing=True,
        accumulate_grad_batches=4,
        profiler="simple",
        min_epochs=50,
    )

    trainer.fit(model=model, datamodule=data_module)

    return model, trainer


if __name__ == "__main__":
    # inception_model, inception_trainer = train_model(
    #     InceptionV3Model, num_classes=5, batch_size=32, max_epochs=100
    # )
    # resnet_model, resnet_trainer = train_model(
    #     ResNet50Model, num_classes=5, batch_size=32, max_epochs=100
    # )
    densenet_model, densenet_trainer = train_model(
        DenseNet121Model, num_classes=5, batch_size=32, max_epochs=20
    )
