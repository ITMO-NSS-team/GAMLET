from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.models import MLP

import automl_surrogate.metrics as metrics_module
from automl_surrogate.data import HeterogeneousBatch
from automl_surrogate.layers.dataset_encoder import DatasetEncoder
from automl_surrogate.models.base import BaseSurrogate


class FusionRankNet(BaseSurrogate):
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        validation_metrics: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__(model_parameters, validation_metrics, lr, weight_decay)
        self.dataset_encoder = DatasetEncoder(**model_parameters["dataset_encoder"])
        self.embedding_joiner = MLP(
            in_channels=self.pipeline_encoder.out_dim + self.dataset_encoder.out_dim,
            hidden_channels=model_parameters["embedding_joiner"]["hidden_channels"],
            out_channels=1,
            num_layers=model_parameters["embedding_joiner"]["num_layers"],
            dropout=model_parameters["embedding_joiner"]["dropout"],
            norm=model_parameters["embedding_joiner"]["norm"],
            plain_last=True,
        )

    def forward(self, heterogen_pipelines: Sequence[HeterogeneousBatch], dataset: Tensor) -> Tensor:
        # [BATCH, N, HIDDEN_1]
        pipelines_embeddings = torch.stack([self.pipeline_encoder(h_p) for h_p in heterogen_pipelines]).permute(1, 0, 2)
        n_pipelines = pipelines_embeddings.shape[1]
        # [BATCH, N, HIDDEN_2]
        dataset_embeddings = self.dataset_encoder(dataset).unsqueeze(1).repeat(1, n_pipelines, 1)
        # [BATCH, N, 1]
        scores = self.embedding_joiner(torch.cat([pipelines_embeddings, dataset_embeddings], dim=-1))
        # [BATCH * N]
        scores = scores.squeeze(-1)
        return scores

    def training_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        # For training arbitrary number of samples N with different metrics is used.
        heterogen_pipelines, dataset, y = batch
        # Sort each pool of candidates in descending order
        indices = y.argsort(dim=1, descending=True)
        # [BATCH, N]
        scores = self.forward(heterogen_pipelines, dataset)
        sorted_scores = scores[torch.arange(scores.shape[0]).unsqueeze(1), indices]
        # [BATCH, N-1]
        difference = sorted_scores[:, :-1] - sorted_scores[:, 1:]
        loss = F.binary_cross_entropy(torch.sigmoid(difference), torch.ones_like(difference))
        self.log("train_loss", loss)
        return loss

    def evaluation_step(
        self,
        prefix: str,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
    ):
        # For train sequnce length is abitrary.
        heterogen_pipelines, dataset, y = batch
        y = torch.softmax(y, dim=1)

        with torch.no_grad():
            scores = self.forward(heterogen_pipelines, dataset)
            scores = torch.softmax(scores, dim=1)

        if "ndcg" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "ndcg")
            self.log(f"{prefix}_ndcg", metric_fn(y.cpu(), scores.cpu()))
        if "precision" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "precision")
            self.log(f"{prefix}_precision", metric_fn(y, scores))
        if "kendalltau" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "kendalltau")
            self.log(f"{prefix}_kendalltau", metric_fn(y.cpu(), scores.cpu()))
