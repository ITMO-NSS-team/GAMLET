"""The module contains general method to train a model from `lib.lightning_modules`."""

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

    lightning_module_class = getattr(lightning_modules, config["lightning_module"]["name"])
    lightning_module = lightning_module_class(**{k: v for k, v in config["lightning_module"].items() if k != "name"})

    logger = TensorBoardLogger(**config["tensorboard_logger"])
    model_checkpoint_callback = ModelCheckpoint(**config["model_checkpoint_callback"])

    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[model_checkpoint_callback, ],
    )
    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lightning_module, dataloaders=test_loader)
