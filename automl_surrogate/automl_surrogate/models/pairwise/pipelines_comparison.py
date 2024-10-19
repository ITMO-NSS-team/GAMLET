from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import automl_surrogate.metrics as metrics_module
from automl_surrogate.data import HeterogeneousBatch
from automl_surrogate.models.base import BaseSurrogate
from automl_surrogate.models.pairwise.bubble_sort import bubble_argsort
from automl_surrogate.models.pairwise.direct_ranker import DirectRanker


class Comparator(BaseSurrogate):
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        validation_metrics: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__(model_parameters, validation_metrics, lr, weight_decay)
        self.comparator = DirectRanker(in_dim=self.pipeline_encoder.out_dim)

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
        # [BATCH, N, HIDDEN]
        pipelines_embeddings = torch.stack([self.pipeline_encoder(h_p) for h_p in heterogen_pipelines]).permute(1, 0, 2)
        sorted_candidates = pipelines_embeddings[:, indices]
        # [BATCH * (N-1), HIDDEN]
        better_candidates = sorted_candidates[:, :-1].flatten(0, 1)
        # [BATCH * (N-1), HIDDEN]
        worse_candidates = sorted_candidates[:, 1:].flatten(0, 1)
        # [BATCH, 1]
        score = self.comparator(better_candidates, worse_candidates).squeeze(1)
        loss = F.mse_loss(score, torch.ones_like(score))
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
            # [BATCH, N, HIDDEN]
            pipelines_embeddings = torch.stack([self.pipeline_encoder(h_p) for h_p in heterogen_pipelines]).permute(
                1, 0, 2
            )
            sorted_indices = bubble_argsort(self.comparator, pipelines_embeddings, self.device)

            seq_len = pipelines_embeddings.shape[1]
            scores = []
            for seq_idxes in sorted_indices:
                seq_scores = torch.empty_like(seq_idxes, dtype=torch.float32)
                seq_scores[seq_idxes] = torch.linspace(0, 1, seq_len, device=seq_scores.device)
                scores.append(seq_scores)
            scores = torch.stack(scores)

        if "ndcg" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "ndcg")
            self.log(f"{prefix}_ndcg", metric_fn(y.cpu(), scores.cpu()))
        if "precision" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "precision")
            self.log(f"{prefix}_precision", metric_fn(y, scores))
        if "kendalltau" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "kendalltau")
            self.log(f"{prefix}_kendalltau", metric_fn(y.cpu(), scores.cpu()))
