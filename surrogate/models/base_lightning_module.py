from typing import Callable

import torch.nn as nn
from pytorch_lightning import LightningModule as LightningModule_
from torch import optim


class BaseLightningModule(LightningModule_):
    def __init__(self, model: nn.Module, loss: Callable, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
