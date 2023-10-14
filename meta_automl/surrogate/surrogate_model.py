from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from sklearn.metrics import ndcg_score
from torch import Tensor
from torch_geometric.data import Batch

from meta_automl.surrogate.encoders import (ColumnDatasetEncoder,
                                            GraphTransformer,
                                            MLPDatasetEncoder,
                                            SimpleGNNEncoder)


def to_labels_k(x, klim):
    """Create y column assigning 1 to first klim elements and 0 to others"""
    vals = np.zeros(len(x))
    if len(x) == 1 or len(x) >= 2 * klim:
        vals[:klim] = 1
    else:
        adjusted_klim = len(x) // 2
        vals[:adjusted_klim] = 1
    return vals


class PipelineDatasetSurrogateModel(LightningModule):
    """Surrogate model to evaluate a pipeline on the given dataset.

    The model is trained to predict a pipeline score.

    Parameters:
    -----------
    model_parameters: Dict of model parameters. The parameters are: TODO.
    lr: Learning rate.
    """

    def __init__(
            self,
            model_parameters: Dict[str, Any],
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            temperature: float = 10,
    ):
        super().__init__()

        if model_parameters['pipe_encoder_type'] == "simple_graph_encoder":
            self.pipeline_encoder = SimpleGNNEncoder(
                **{k: v for k, v in model_parameters.items() if k != "name"})
        elif model_parameters['pipe_encoder_type'] == "graph_transformer":
            self.pipeline_encoder = GraphTransformer(
                **{k: v for k, v in model_parameters.items() if k != "name"})

        if model_parameters['dataset_encoder_type'] == "column":
            self.dataset_encoder = ColumnDatasetEncoder(
                model_parameters['dim_dataset'],
                hidden_dim=model_parameters['d_model_dset'],
                output_dim=model_parameters['d_model_dset'],
            )
        elif model_parameters['dataset_encoder_type'] == "aggregated":
            self.dataset_encoder = MLPDatasetEncoder(
                model_parameters['dim_dataset'],
                hidden_dim=model_parameters['d_model_dset'],
                output_dim=model_parameters['d_model_dset'],
            )
        else:
            raise ValueError("dataset_encoder_type should be 'column' or 'aggregated'")

        cat_dim = self.dataset_encoder.dim + self.pipeline_encoder.dim
        self.final_model = nn.Sequential(
            nn.BatchNorm1d(cat_dim),
            nn.Linear(cat_dim, cat_dim),
            nn.ReLU(),
            nn.Linear(cat_dim, 1),
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature

        # Migration to pytorch_lightning > 1.9.5
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()  # TODO: is it required? We have config file.

        self.K_TOP = 3
        self.to_labels = partial(to_labels_k, klim=self.K_TOP)

    def forward(self, x_graph: Batch, dset) -> Tensor:
        """Computation method.

        Parameters:
        -----------
        x_graph: Graph data.
        x_dset: Dataset data.

        Returns:
        --------
        A pipeline score.
        """
        if not x_graph.edge_index.shape[0]:
            x_graph.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        x_graph.x = x_graph.x.view(-1)

        z_pipeline = self.pipeline_encoder(x_graph)
        z_dataset = self.dataset_encoder(dset)
     
        assert not torch.isnan(z_dataset).any()
    
        return self.final_model(torch.cat((z_pipeline, z_dataset), 1))

    def training_step(self, batch: Tuple[Tensor, Batch, Tensor, Batch, Tensor], *args, **kwargs: Any) -> Tensor:
        """Training step.

        Parameters:
        -----------
        batch: A tuple of:
        * `task_id`: Task ID (a.k.a dataset ID),
        * `pipe_id`: Pipeline ID,
        * `x_graph`: Graph data,
        * `x_dset`: Dataset data,
        * `y_true`: Pipeline score.

        Returns:
        --------
        Loss value.
        """
        task_id, pipe_id, x_graph, x_dset, y_true = batch
        y_pred = self.forward(x_graph, x_dset)
        y_pred = torch.squeeze(y_pred)
        loss = F.mse_loss(torch.squeeze(y_pred), y_true)
        self.log("train_loss", loss, batch_size=y_true.shape[0])
        return loss

    def validation_step(self, batch: Tuple[Tensor, Batch, Tensor, Batch, Tensor], *args, **kwargs: Any) -> None:
        """Validation step.

        Parameters:
        -----------
        batch: A tuple of:
        * `task_id`: Task ID (a.k.a dataset ID),
        * `pipe_id`: Pipeline ID,
        * `x_graph`: Graph data,
        * `x_dset`: Dataset data,
        * `y_true`: Pipeline score.
        """

        task_id, pipe_id, x_graph, x_dset, y_true = batch
        y_pred = self.forward(x_graph, x_dset)
        y_pred = torch.squeeze(y_pred)
        output = {
            'task_id': task_id,
            'pipe_id': pipe_id.cpu().numpy(),
            'y_pred': y_pred.detach().cpu().numpy(),
            'y_true': y_true.detach().cpu().numpy(),
        }
        self.validation_step_outputs.append(output)

    def test_step(self, batch: Tuple[Tensor, Batch, Tensor, Batch, Tensor], *args, **kwargs: Any) -> None:
        """Test step.

        Parameters:
        -----------
        batch: A tuple of:
        * `task_id`: Task ID (a.k.a dataset ID),
        * `pipe_id`: Pipeline ID,
        * `x_graph`: Graph data,
        * `x_dset`: Dataset data,
        * `y_true`: Pipeline score.
        """
        task_id, pipe_id, x_graph, x_dset, y_true = batch
        y_pred = self.forward(x_graph, x_dset)
        y_pred = torch.squeeze(y_pred)
        output = {
            'task_id': task_id,
            'pipe_id': pipe_id.cpu().numpy(),
            'y_pred': y_pred.detach().cpu().numpy(),
            'y_true': y_true.detach().cpu().numpy(),
        }
        self.test_step_outputs.append(output)

    def _get_metrics(self, outputs: Dict[str, np.ndarray]) -> Any:
        """Calculate mean metrics score value over multiple predictions.

        Parameters:
        -----------
        outputs: storage of information to calculate NDCG score.

        Returns:
        --------
        Mean metrics score.
        """

        def gr_calc(inp):
            kk = 3
            y_true = self.to_labels(inp['y_true'].values).reshape(1, -1)
            y_pred = inp['y_pred'].values.reshape(1, -1)

            res = {}
            # MRR
            idx_y_pred_sorted = np.argsort(y_pred.flatten())[::-1]
            mask = y_true.flatten()[idx_y_pred_sorted] > 0
            rank_max = idx_y_pred_sorted[mask][0]
            res['mrr'] = 1. / (rank_max + 1)
            # NDCG
            res['ndcg'] = ndcg_score(y_true, y_pred, k=10)
            # HITS

            res['hits'] = mask[:kk].sum() / kk
            return pd.Series(res, index=['ndcg', 'hits', 'mrr'])

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
        df = df.sort_values(by='y_true', ascending=False)
        res = df.groupby('task_id').apply(gr_calc)
        return res.mean().to_dict()

    def on_validation_epoch_end(self) -> None:
        """Calculate NDCG score over predicted during validation pipeline estimates."""
        ndcg_mean = self._get_metrics(self.validation_step_outputs)['ndcg']
        print("val_ndcg = ", ndcg_mean)
        self.log("val_ndcg", ndcg_mean)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """Calculate NDCG score over predicted during testing pipeline estimates."""
        metrics = self._get_metrics(self.test_step_outputs)
        self.log("test_ndcg", metrics['ndcg'])
        self.log("test_mrr", metrics['mrr'])
        self.log("test_hits", metrics['hits'])
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(list(self.pipeline_encoder.parameters())
                                + list(self.dataset_encoder.parameters())
                                + list(self.final_model.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay)
        return optimizer


class RankingPipelineDatasetSurrogateModel(PipelineDatasetSurrogateModel):
    """Surrogate model to evaluate a pipeline on the given dataset.

    The model is trained to rank between two pipelines.
    The loss is calculated as the ranknet loss.

    Parameters:
    -----------
    model_parameters: Dict of model parameters. The parameters are: TODO.
    loss_name: Loss name from torch.nn.functional. Default: `None`.
        If the parameter is `None`, one should implement `self.loss` method in a subclass.
    lr: Learning rate.
    """
#     def loss(self, score1: Tensor, score2: Tensor, target: Tensor) -> Tensor:
#         """Ranknet loss.

#         Parameters:
#         -----------
#         score1: Predicted score of the first pipeline.
#         score2: Predicted score of the second pipeline.
#         target: Target value.

#         Returns:
#         Loss value.
#         """

#         o = torch.sigmoid(score1 - score2)
#         loss = (-target * o + F.softplus(o)).mean()
#         return loss

    def training_step(self, batch: Tuple[Tensor, Batch, Tensor, Batch, Tensor], *args, **kwargs: Any) -> Tensor:
        """Training step.

        Parameters:
        -----------
        batch: A tuple of:
        * `task_id`: Task ID (a.k.a dataset ID),
        * `pipe_id`: Pipeline ID,
        * `x_graph`: Graph data,
        * `x_dset`: Dataset data,
        * `y_true`: Pipeline score.

        Returns:
        --------
        Loss value.
        """
        x_pipe1, x_pipe2, dset_data, y = batch

        pred1 = torch.squeeze(self.forward(x_pipe1, dset_data))
        pred2 = torch.squeeze(self.forward(x_pipe2, dset_data))
        loss = F.binary_cross_entropy_with_logits((pred1 - pred2) * self.temperature, y)
        # loss = self.loss(pred1, pred2, y)
        self.log("train_loss", loss)
        return loss
