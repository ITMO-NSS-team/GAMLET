"""The module contains general method to train a model from `surrogate.models`."""

from typing import Dict, Any

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from surrogate import datasets, models


def train(config: Dict[str, Any]):
    train_dataset = getattr(datasets, config["train_dataset"]["name"])(**config["train_dataset"]["parameters"])
    val_dataset = getattr(datasets, config["val_dataset"]["name"])(**config["val_dataset"]["parameters"])
    test_dataset = getattr(datasets, config["test_dataset"]["name"])(**config["test_dataset"]["parameters"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    model_class = getattr(models, config["model"]["name"])
    model = model_class(**{k: v for k, v in config["model"].items() if k != "name"})

    logger = TensorBoardLogger(**config["tensorboard_logger"]) if config["tensorboard_logger"] is not None else None
    model_checkpoint_callback = ModelCheckpoint(**config["model_checkpoint_callback"])

    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[model_checkpoint_callback, ],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)
