from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_activation_fn

import automl_surrogate.metrics as metrics_module
from automl_surrogate.data import HeterogeneousBatch
from automl_surrogate.layers.dataset_encoder import DatasetEncoder
from automl_surrogate.models.base import BaseSurrogate
from automl_surrogate.models.listwise.set_rank import SetRank


class BaseDataAwareRanker(BaseSurrogate):
    def training_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor, Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        heterogen_pipelines, dataset, y = batch
        scores = self.forward(heterogen_pipelines, dataset)
        loss = F.kl_div(
            torch.log_softmax(scores, dim=1),
            torch.log_softmax(y, dim=1),
            log_target=True,
        )
        self.log("train_loss", loss)
        return loss

    def evaluation_step(
        self,
        prefix: str,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
    ):
        heterogen_pipelines, dataset, y = batch
        y = torch.softmax(y, dim=1)

        with torch.no_grad():
            scores = self.forward(heterogen_pipelines, dataset)

        if "ndcg" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "ndcg")
            self.log(f"{prefix}_ndcg", metric_fn(y.cpu(), scores.cpu()))
        if "precision" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "precision")
            self.log(f"{prefix}_precision", metric_fn(y, scores))
        if "kendalltau" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "kendalltau")
            self.log(f"{prefix}_kendalltau", metric_fn(y.cpu(), scores.cpu()))


class CrossAttentionTransformerEncoder(nn.TransformerEncoder):
    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        for mod in self.layers:
            output = mod(q, kv)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CrossAttentionTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        kdim: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(nn.TransformerEncoderLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, kdim=kdim, vdim=kdim, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        x = self.norm1(q + self._ca_block(q, kv))
        x = self.norm2(x + self._ff_block(x))
        return x

    # cross-attention block
    def _ca_block(self, query: Tensor, kv: Tensor) -> Tensor:
        # Expected input shapes are [Batch, N, HIDDEN_1], [Batch, HIDDEN_2]
        kv = kv.unsqueeze(1)
        x = self.cross_attn(query, kv, kv, need_weights=False)[0]
        return self.dropout1(x)


class FusionSetRank(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nhead: int,
        dim_feedforward: int,
        dropout: int,
        num_layers: int,
        mhca_block_params: dict,
        dataset_dim: int,
    ):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.mhsa_block = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )
        self.mhca_block = self.build_mhca_block(
            in_dim,
            dataset_dim,
            mhca_block_params,
        )
        self.linear = nn.Linear(in_dim, 1)
        self.out_dim = 1

    @staticmethod
    def build_mhca_block(qdim: int, kdim: int, config: Dict[str, Any]) -> nn.TransformerEncoder:
        transformer_layer = CrossAttentionTransformerEncoderLayer(
            d_model=qdim,
            kdim=kdim,
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        mhca_block = CrossAttentionTransformerEncoder(
            transformer_layer,
            num_layers=config["num_layers"],
        )
        mhca_block.out_dim = qdim
        return mhca_block

    def forward(self, pipelines_embeddings: Tensor, dataset_embeddings: Tensor) -> Tensor:
        # Expected input shape is [BATCH, N, HIDDEN]
        mhsa_output = self.mhsa_block(pipelines_embeddings)
        reweighted = self.mhca_block(mhsa_output, dataset_embeddings)
        scores = self.linear(reweighted).squeeze(2)
        # Output shape is [BATCH, N]
        return scores


class LateFusionRanker(BaseDataAwareRanker):
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        validation_metrics: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__(model_parameters, validation_metrics, lr, weight_decay)
        self.dataset_encoder = DatasetEncoder(**model_parameters["dataset_encoder"])
        self.set_rank = FusionSetRank(
            in_dim=self.pipeline_encoder.out_dim,
            **model_parameters["set_rank"],
            mhca_block_params=model_parameters["mhca_block"],
            dataset_dim=self.dataset_encoder.out_dim,
        )

    def forward(self, heterogen_pipelines: Sequence[HeterogeneousBatch], dataset: Tensor) -> Tensor:
        # [BATCH, N, HIDDEN_1]
        pipelines_embeddings = torch.stack([self.pipeline_encoder(h_p) for h_p in heterogen_pipelines]).permute(1, 0, 2)
        # [BATCH, HIDDEN_2]
        dataset_embeddings = self.dataset_encoder(dataset)
        # [BATCH, N]
        scores = self.set_rank(pipelines_embeddings, dataset_embeddings)
        # [BATCH, N]
        return scores


class EarlyFusionRanker(BaseDataAwareRanker):
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        validation_metrics: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__(model_parameters, validation_metrics, lr, weight_decay)
        self.dataset_encoder = DatasetEncoder(**model_parameters["dataset_encoder"])
        self.set_rank = SetRank(
            in_dim=self.pipeline_encoder.out_dim + self.dataset_encoder.out_dim,
            **model_parameters["set_rank"],
        )

    def forward(self, heterogen_pipelines: Sequence[HeterogeneousBatch], dataset: Tensor) -> Tensor:
        # [BATCH, N, HIDDEN_1]
        pipelines_embeddings = torch.stack([self.pipeline_encoder(h_p) for h_p in heterogen_pipelines]).permute(1, 0, 2)
        n_pipelines = pipelines_embeddings.shape[1]
        # [BATCH, N, HIDDEN_2]
        dataset_embeddings = self.dataset_encoder(dataset).unsqueeze(1).repeat(1, n_pipelines, 1)
        # [BATCH, N, HIDDEN_1 + HIDDEN_2]
        joined_embeddings = torch.cat([pipelines_embeddings, dataset_embeddings], dim=-1)
        # [BATCH, N]
        scores = self.set_rank(joined_embeddings)
        return scores
