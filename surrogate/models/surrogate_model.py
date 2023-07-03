from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from surrogate.encoders import GraphTransformer, MLPDatasetEncoder
from sklearn.metrics import top_k_accuracy_score, average_precision_score, ndcg_score


def gr_ndcg(inp):
    y1 = inp['y_true'].values.reshape(1, -1)
    y2 = inp['y_pred'].values.reshape(1, -1)
    return ndcg_score(y1, y2, k=3)

def ranknet_loss(s1, s2, t):
    o = torch.sigmoid(s1 - s2)
    loss = (-t * o + F.softplus(o)).mean()
    return loss

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
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, x_graph,  x_dset):
        z_pipeline = self.pipeline_encoder(x_graph)
        z_dataset = self.dataset_encoder(x_dset)
        return self.final_model(torch.cat((z_pipeline, z_dataset), 1))

    def training_step(self, batch, batch_idx):
        task_id, pipe_id, x_graph, x_dset, y_true = batch
        y_pred = self.forward(x_graph, x_dset)
        y_pred = torch.squeeze(y_pred)
        loss = self.loss(torch.squeeze(y_pred), y_true)
        self.log("train_loss", loss, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        task_id, pipe_id, x_graph, x_dset, y_true = batch
        y_pred = self.forward(x_graph, x_dset)
        y_pred = torch.squeeze(y_pred)
        output = {'task_id': task_id.cpu().numpy(),
                'pipe_id':pipe_id.cpu().numpy(),
                'y_pred':y_pred.detach().cpu().numpy(),
                'y_true':y_true.detach().cpu().numpy()}
        self.validation_step_outputs.append(output)


    def test_step(self, batch, batch_idx):
        task_id, pipe_id, x_graph, x_dset, y_true = batch
        y_pred = self.forward(x_graph, x_dset)
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
        # Remove groups with single element to enable work of sklearn.metrics.ndcg
        ndcg_mean = df.groupby('task_id').filter(lambda x: len(x) > 1).groupby('task_id').apply(gr_ndcg).mean()
        return ndcg_mean

    def on_validation_epoch_end(self):
        ndcg_mean = self._get_ndcg(self.validation_step_outputs)
        self.log("val_ndcg", ndcg_mean)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        ndcg_mean = self._get_ndcg(self.test_step_outputs)

        task_ids, pipe_ids, y_preds, y_trues = [], [], [], []
        for output in self.test_step_outputs:
            y_preds.append(output['y_pred'])
            y_trues.append(output['y_true'])
        y_true = np.concatenate(y_trues)
        y_score = np.concatenate(y_preds)


        self.log("test_ndcg", ndcg_mean)
        self.log("test_mrr", average_precision_score(y_true, y_score))
        self.log("test_hits", top_k_accuracy_score(y_true, y_score, k=1))
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.pipeline_encoder.parameters()) +
                                      list(self.dataset_encoder.parameters()) +
                                      list(self.final_model.parameters()),
                                      lr=self.lr,
                                      weight_decay=1e-5)
        return optimizer

class RankingSurrogateModel(SurrogateModel):
    def training_step(self, batch, batch_idx):
        x_pipe1, x_dset1, x_pipe2, x_dset2, y = batch

        pred1 = torch.squeeze(self.forward(x_pipe1, x_dset1))
        pred2 = torch.squeeze(self.forward(x_pipe2, x_dset2))

        loss = ranknet_loss(pred1, pred2, y)
        self.log("train_loss", loss)# batch_size=batch[0].shape[0])
        return loss

class SurrogateModelNoMeta(SurrogateModel):
    def __init__(
            self,
            model_parameters: Dict[str, Any],
            loss_name: str,
            lr: float = 1e-3,
    ):
        """loss_name: loss name from torch.nn.functional"""
        LightningModule.__init__(self)

        self.pipeline_encoder = GraphTransformer(
            **{k: v for k, v in model_parameters.items() if k != "name"})

        self.final_model = nn.Sequential(
            nn.BatchNorm1d(model_parameters['d_model']),
            nn.Linear(
                model_parameters['d_model'], model_parameters['d_model'] * 2),
            nn.ReLU(),
            nn.Linear(model_parameters['d_model'] * 2, 1),
        )

        self.loss = getattr(F, loss_name)
        self.lr = lr

        # Migration to pytorch_lightning > 1.9.5
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, x_graph,  x_dset):
        z_pipeline = self.pipeline_encoder(x_graph)
        return self.final_model(z_pipeline)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.pipeline_encoder.parameters()) +
                                      list(self.final_model.parameters()),
                                      lr=self.lr,
                                      weight_decay=1e-5)
        return optimizer

class RankingSurrogateModelNoMeta(SurrogateModelNoMeta):
    def training_step(self, batch, batch_idx):
        x_pipe1, x_dset1, x_pipe2, x_dset2, y = batch

        pred1 = torch.squeeze(self.forward(x_pipe1, x_dset1))
        pred2 = torch.squeeze(self.forward(x_pipe2, x_dset2))

        loss = ranknet_loss(pred1, pred2, y)
        self.log("train_loss", loss)# batch_size=batch[0].shape[0])
        return loss