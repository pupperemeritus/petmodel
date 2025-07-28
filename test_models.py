import os

import pandas as pd
import pytorch_lightning as pl
import torch
from IPython import display

from dataloader import PETDataModule
from model import GuruNet
from other_models import DenseNet121Model, InceptionV3Model, ResNet50Model


def get_latest_checkpoint(model_class_name):
    checkpoints_dir = f"checkpoints/{model_class_name}"
    versions = [d for d in os.listdir(checkpoints_dir) if d.startswith("version_")]
    latest_version = max(versions, key=lambda x: int(x.split("_")[1]))

    checkpoint_dir = os.path.join(checkpoints_dir, latest_version)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    best_checkpoint = min(
        checkpoints, key=lambda x: float(x.split("-")[-2].split("=")[-1])
    )

    return os.path.join(checkpoint_dir, best_checkpoint)


def test_model(model, data_module: pl.LightningDataModule, trainer: pl.Trainer):
    # Load the best checkpoint
    best_checkpoint_path = get_latest_checkpoint(model.__name__)

    # Load the model from checkpoint
    model = model.load_from_checkpoint(best_checkpoint_path)

    # Test the model
    test_results = trainer.test(model=model, datamodule=data_module, verbose=False)
    print(f"{model.__class__.__name__}:")
    display.display(pd.DataFrame(test_results))


if __name__ == "__main__":
    models = [GuruNet, DenseNet121Model, ResNet50Model, InceptionV3Model]
    data_module = PETDataModule()
    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="16-mixed",
        logger=None,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir="./test_logs",
    )
    for model in models:
        test_model(model, data_module, trainer)
