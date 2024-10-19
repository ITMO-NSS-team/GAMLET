from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import automl_surrogate.metrics as metrics_module
from automl_surrogate.data import HeterogeneousBatch
from automl_surrogate.models.base import BaseSurrogate


class RankNet(BaseSurrogate):
    # Same hypothesis as in https://arxiv.org/pdf/1912.05891.pdf
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        validation_metrics: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__(model_parameters, validation_metrics, lr, weight_decay)
        self.linear = nn.Linear(self.pipeline_encoder.out_dim, 1)

    def forward(self, heterogen_pipelines: Sequence[HeterogeneousBatch]) -> Tensor:
        # [BATCH, N, HIDDEN]
        pipelines_embeddings = torch.stack([self.pipeline_encoder(h_p) for h_p in heterogen_pipelines]).permute(1, 0, 2)
        # [BATCH, N]
        scores = self.linear(pipelines_embeddings).squeeze(2)
        return scores

    def training_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        # For training arbitrary number of samples N with different metrics is used.
        heterogen_pipelines, y = batch
        # Sort each pool of candidates in descending order
        indices = y.argsort(dim=1, descending=True)
        # [BATCH, N]
        scores = self.forward(heterogen_pipelines)
        sorted_scores = scores[:, indices]
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
        heterogen_pipelines, y = batch
        y = torch.softmax(y, dim=1)

        with torch.no_grad():
            scores = self.forward(heterogen_pipelines)
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
