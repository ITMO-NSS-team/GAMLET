from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from sklearn.metrics import average_precision_score, ndcg_score, top_k_accuracy_score
from torch import Tensor
from torch_geometric.data import Batch

from meta_automl.surrogate.encoders import GraphTransformer, MLPDatasetEncoder


class PipelineDatasetSurrogateModel(LightningModule):
    """Surrogate model to evaluate a pipeline on the given dataset.

    The model is trained to predict a pipeline score.

    Parameters:
    -----------
    model_parameters: Dict of model parameters. The parameters are: TODO.
    loss_name: Loss name from torch.nn.functional. Default: `None`.
        If the parameter is `None`, one should implement `self.loss` method in a subclass.
    lr: Learning rate.
    """

    def __init__(
            self,
            model_parameters: Dict[str, Any],
            loss_name: Optional[str] = None,
            lr: float = 1e-3,
    ):
        super().__init__()

        self.pipeline_encoder = GraphTransformer(
            **{k: v for k, v in model_parameters.items() if k != "name"})

        self.dataset_encoder = MLPDatasetEncoder(
            model_parameters['dim_dataset'],
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

        if loss_name is not None:
            assert_message = f"Class`{type(self)}` has custom loss. Do not pass `loss_name` argument."
            assert hasattr(self, "loss"), assert_message
            self.loss = getattr(F, loss_name)
        else:
            assert_message = "One should implement a loss function in a subclass"\
                             " or provide a loss name from `torch.nn.functional`."
            assert hasattr(self, "loss"), assert_message

        self.lr = lr

        # Migration to pytorch_lightning > 1.9.5
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()  # TODO: is it required? We have config file.

    def forward(self, x_graph: Batch, x_dset: Tensor) -> Tensor:
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
            x_graph.edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        x_graph.x = x_graph.x.view(-1)
            
        z_pipeline = self.pipeline_encoder(x_graph)
        z_dataset = self.dataset_encoder(x_dset)
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

        _, _, x_graph, x_dset, y_true = batch
        y_pred = self.forward(x_graph, x_dset)
        y_pred = torch.squeeze(y_pred)
        loss = self.loss(torch.squeeze(y_pred), y_true)
        self.log("train_loss", loss, batch_size=batch[0].shape[0])
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
        output = {'task_id': task_id.cpu().numpy(),
                'pipe_id':pipe_id.cpu().numpy(),
                'y_pred':y_pred.detach().cpu().numpy(),
                'y_true':y_true.detach().cpu().numpy()}
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
            'task_id': task_id.cpu().numpy(),
            'pipe_id': pipe_id.cpu().numpy(),
            'y_pred': y_pred.detach().cpu().numpy(),
            'y_true': y_true.detach().cpu().numpy(),
        }
        self.test_step_outputs.append(output)

    def _get_ndcg(self, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate mean NDCG score value over multiple predictions.

        Parameters:
        -----------
        outputs: storage of information to calculate NDCG score.

        Returns:
        --------
        Mean NDCG score.
        """

        def gr_ndcg(inp):
            y1 = inp['y_true'].values.reshape(1, -1)
            y2 = inp['y_pred'].values.reshape(1, -1)
            return ndcg_score(y1, y2, k=3)

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

    def on_validation_epoch_end(self) -> None:
        """Calculate NDCG score over predicted during validation pipeline estimates."""
        ndcg_mean = self._get_ndcg(self.validation_step_outputs)
        self.log("val_ndcg", ndcg_mean)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """Calculate NDCG score over predicted during testing pipeline estimates."""
        ndcg_mean = self._get_ndcg(self.test_step_outputs)

        y_preds, y_trues = [], []
        for output in self.test_step_outputs:
            y_preds.append(output['y_pred'])
            y_trues.append(output['y_true'])
        y_true = np.concatenate(y_trues)
        y_score = np.concatenate(y_preds)

        self.log("test_ndcg", ndcg_mean)
        self.log("test_mrr", average_precision_score(y_true, y_score))
        self.log("test_hits", top_k_accuracy_score(y_true, y_score, k=1))
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(list(self.pipeline_encoder.parameters()) +
                                      list(self.dataset_encoder.parameters()) +
                                      list(self.final_model.parameters()),
                                      lr=self.lr,
                                      weight_decay=1e-5)
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
    def loss(self, score1: Tensor, score2: Tensor, target: Tensor) -> Tensor:
        """Ranknet loss.

        Parameters:
        -----------
        score1: Predicted score of the first pipeline.
        score2: Predicted score of the second pipeline.
        target: Target value.

        Returns:
        Loss value.
        """

        o = torch.sigmoid(score1 - score2)
        loss = (-target * o + F.softplus(o)).mean()
        return loss

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

        x_pipe1, x_dset1, x_pipe2, x_dset2, y = batch
        pred1 = torch.squeeze(self.forward(x_pipe1, x_dset1))
        pred2 = torch.squeeze(self.forward(x_pipe2, x_dset2))
        loss = self.loss(pred1, pred2, y)
        self.log("train_loss", loss)
        return loss
