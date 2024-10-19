from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.models import MLP

import automl_surrogate.metrics as metrics_module
from automl_surrogate.data import HeterogeneousBatch
from automl_surrogate.layers.dataset_encoder import DatasetEncoder
from automl_surrogate.models.base import BaseSurrogate
from automl_surrogate.models.pairwise.bubble_sort import bubble_argsort
from automl_surrogate.models.pairwise.direct_ranker import DirectRanker, FusionDirectRanker


class EarlyFusionComparator(BaseSurrogate):
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
            out_channels=model_parameters["embedding_joiner"]["hidden_channels"],
            num_layers=model_parameters["embedding_joiner"]["num_layers"],
            dropout=model_parameters["embedding_joiner"]["dropout"],
            norm=model_parameters["embedding_joiner"]["norm"],
            bias=[True] * (model_parameters["embedding_joiner"]["num_layers"] - 1) + [False],
        )
        self.embedding_joiner.out_dim = self.embedding_joiner.out_channels
        self.comparator = DirectRanker(in_dim=self.embedding_joiner.out_dim)

    def embed_inputs(self, heterogen_pipelines: Sequence[HeterogeneousBatch], dataset: Tensor) -> Tensor:
        # [BATCH, N, HIDDEN_1]
        pipelines_embeddings = torch.stack([self.pipeline_encoder(h_p) for h_p in heterogen_pipelines]).permute(1, 0, 2)
        n_pipelines = pipelines_embeddings.shape[1]
        # [BATCH, N, HIDDEN_2]
        dataset_embeddings = self.dataset_encoder(dataset).unsqueeze(1).repeat(1, n_pipelines, 1)
        # [BATCH, N, HIDDEN_3]
        joined_embeddings = self.embedding_joiner(torch.cat([pipelines_embeddings, dataset_embeddings], dim=-1))
        return joined_embeddings

    def training_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor, Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        # For training arbitrary number of samples N with different metrics is used.
        heterogen_pipelines, dataset, y = batch
        # Sort each pool of candidates in descending order
        indices = y.argsort(dim=1, descending=True)
        # [BATCH, N, HIDDEN_3]
        joined_embeddings = self.embed_inputs(heterogen_pipelines, dataset)
        sorted_candidates = joined_embeddings[torch.arange(joined_embeddings.size(0)).unsqueeze(1), indices]
        # [BATCH * (N-1), HIDDEN]
        better_candidates = sorted_candidates[:, :-1].flatten(0, 1)
        # [BATCH * (N-1), HIDDEN]
        worse_candidates = sorted_candidates[:, 1:].flatten(0, 1)
        # [BATCH, 1]
        score = self.comparator(better_candidates, worse_candidates).squeeze(1)
        loss = F.mse_loss(score, torch.ones_like(score))
        # Maybe try same learning as LateFusion?
        # score_forward = self.comparator(better_candidates, worse_candidates).squeeze(1)
        # score_reversed = self.comparator(worse_candidates, better_candidates).squeeze(1)
        # loss_forward = F.mse_loss(score_forward, torch.full_like(score_forward, fill_value=1.))
        # loss_reversed = F.mse_loss(score_reversed, torch.full_like(score_reversed, fill_value=-1.))
        # loss = (loss_forward + loss_reversed) / 2
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
            # [BATCH, N, HIDDEN_3]
            joined_embeddings = self.embed_inputs(heterogen_pipelines, dataset)
            sorted_indices = bubble_argsort(self.comparator, joined_embeddings, self.device)

            seq_len = joined_embeddings.shape[1]
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


class LateFusionComparator(BaseSurrogate):
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        validation_metrics: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__(model_parameters, validation_metrics, lr, weight_decay)
        self.dataset_encoder = DatasetEncoder(**model_parameters["dataset_encoder"])
        self.comparator = FusionDirectRanker(in_dim=self.pipeline_encoder.out_dim + self.dataset_encoder.out_dim)

    def training_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor, Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        # For training arbitrary number of samples N with different metrics is used.
        heterogen_pipelines, dataset, y = batch
        n_pipelines = len(heterogen_pipelines)
        # Sort each pool of candidates in descending order
        indices = y.argsort(dim=1, descending=True)
        # [BATCH, N, HIDDEN_1]
        pipelines_embeddings = torch.stack([self.pipeline_encoder(h_p) for h_p in heterogen_pipelines]).permute(1, 0, 2)
        # [BATCH, N, HIDDEN_2]
        dataset_embeddings = self.dataset_encoder(dataset)

        sorted_candidates = pipelines_embeddings[torch.arange(pipelines_embeddings.size(0)).unsqueeze(1), indices]
        # [BATCH * (N-1), HIDDEN]
        better_candidates = sorted_candidates[:, :-1].flatten(0, 1)
        # [BATCH * (N-1), HIDDEN]
        worse_candidates = sorted_candidates[:, 1:].flatten(0, 1)
        # [BATCH, 1]
        dataset_embeddings = dataset_embeddings.unsqueeze(1).repeat(1, n_pipelines - 1, 1).flatten(0, 1)
        score_forward = self.comparator(better_candidates, worse_candidates, dataset_embeddings).squeeze(1)
        score_reversed = self.comparator(worse_candidates, better_candidates, dataset_embeddings).squeeze(1)
        loss_forward = F.mse_loss(score_forward, torch.full_like(score_forward, fill_value=1.0))
        loss_reversed = F.mse_loss(score_reversed, torch.full_like(score_reversed, fill_value=-1.0))
        loss = (loss_forward + loss_reversed) / 2
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
            # [BATCH, N, HIDDEN_1]
            pipelines_embeddings = torch.stack([self.pipeline_encoder(h_p) for h_p in heterogen_pipelines]).permute(
                1, 0, 2
            )
            # [BATCH, HIDDEN_2]
            dataset_embeddings = self.dataset_encoder(dataset)
            sorted_indices = bubble_argsort(self.comparator, pipelines_embeddings, self.device, dataset_embeddings)
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
