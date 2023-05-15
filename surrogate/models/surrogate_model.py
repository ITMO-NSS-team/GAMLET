from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from surrogate.encoders import GraphTransformer, MLPDatasetEncoder


def gr_ndcg(inp):
    y1 = inp['y_true'].values.reshape(1, -1)
    y2 = inp['y_pred'].values.reshape(1, -1)
    return ndcg_score(y1, y2, k=3)


class SurrogateModel(LightningModule):
    def __init__(
            self,
            model_parameters: Dict[str, Any],
            loss_name: str,
            lr: float = 1e-3,
    ):
        """loss_name: loss name from torch.nn.functional"""
        super().__init__()

        self.pipeline_encoder = GraphTransformer(
            **{k: v for k, v in model_parameters.items() if k != "name"})
        self.dataset_encoder = MLPDatasetEncoder(
            model_parameters['dim_dataset'],
            model_parameters['meta_data'],
            hidden_dim=model_parameters['d_model'],
            output_dim=model_parameters['d_model'],
        )

        self.final_model = nn.Sequential(
            nn.BatchNorm1d(model_parameters['d_model'] * 2),
            nn.Linear(
                model_parameters['d_model'] * 2, model_parameters['d_model'] * 2),
            nn.ReLU(),
            nn.Linear(model_parameters['d_model'] * 2, 1),
        )

        self.loss = getattr(F, loss_name)
        self.lr = lr

        # Migration to pytorch_lightning > 1.9.5
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        task_id, pipe_id, x_graph, x_dset, y = data
        z_pipeline = self.pipeline_encoder(x_graph)
        z_dataset = self.dataset_encoder(x_dset)
        return task_id, pipe_id, self.final_model(torch.cat((z_pipeline, z_dataset), 1)), y

    def training_step(self, batch, batch_idx):
        task_id, pipe_id, y_pred, y_true = self.forward(batch)
        y_pred = torch.squeeze(y_pred)
        loss = self.loss(torch.squeeze(y_pred), y_true)
        self.log("train_loss", loss, batch_size=y_true.shape[0])
        output = {
            'task_id': task_id.cpu().numpy(),
            'pipe_id': pipe_id.cpu().numpy(),
            'y_pred': y_pred.detach().cpu().numpy(),
            'y_true': y_true.detach().cpu().numpy(),
        }
        self.train_step_outputs.append(output)
        return loss

    def validation_step(self, batch, batch_idx):
        task_id, pipe_id, y_pred, y_true = self.forward(batch)
        y_pred = torch.squeeze(y_pred)
        output = {
            'task_id': task_id.cpu().numpy(),
            'pipe_id': pipe_id.cpu().numpy(),
            'y_pred': y_pred.detach().cpu().numpy(),
            'y_true': y_true.detach().cpu().numpy(),
        }
        self.validation_step_outputs.append(output)

    def test_step(self, batch, batch_idx):
        task_id, pipe_id, y_pred, y_true = self.forward(batch)
        y_pred = torch.squeeze(y_pred)
        output = {
            'task_id': task_id.cpu().numpy(),
            'pipe_id': pipe_id.cpu().numpy(),
            'y_pred': y_pred.detach().cpu().numpy(),
            'y_true': y_true.detach().cpu().numpy(),
        }
        self.test_step_outputs.append(output)

    def _get_ndcg(self, outputs):
        task_ids, pipe_ids, y_preds, y_trues = [], [], [], []
        for output in outputs:
            task_ids.append(output['task_id'])
            pipe_ids.append(output['pipe_id'])
            y_preds.append(output['y_pred'])
            y_trues.append(output['y_true'])

        df = pd.DataFrame({'task_id': np.concatenate(task_ids),
                           'pipe_id': np.concatenate(pipe_ids),
                           'y_pred': np.concatenate(y_preds),
                           'y_true': np.concatenate(y_trues)})
        ndcg_mean = df.groupby('task_id').apply(gr_ndcg).mean()
        return ndcg_mean

    def on_train_epoch_end(self):
        ndcg_mean = self._get_ndcg(self.train_step_outputs)
        self.log("train_ndcg", ndcg_mean)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        ndcg_mean = self._get_ndcg(self.validation_step_outputs)
        self.log("val_ndcg", ndcg_mean)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        ndcg_mean = self._get_ndcg(self.test_step_outputs)
        self.log("test_ndcg", ndcg_mean)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.pipeline_encoder.parameters()) +
                                      list(self.dataset_encoder.parameters()) +
                                      list(self.final_model.parameters()),
                                      lr=self.lr,
                                      weight_decay=1e-5)
        return optimizer
