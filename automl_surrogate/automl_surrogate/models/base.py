from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Batch

import automl_surrogate.losses as losses_module
import automl_surrogate.metrics as metrics_module
from automl_surrogate.data import HeterogeneousBatch
from automl_surrogate.layers.encoders import GraphTransformer, SimpleGNNEncoder
from automl_surrogate.layers.pipeline_encoder import PipelineEncoder
from automl_surrogate.models.node_homogenizer import build_node_homogenizer


class BaseSurrogate(LightningModule):
    # Implements a pipeline encoding and model training.
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        validation_metrics: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_metrics = validation_metrics
        self.pipeline_encoder = PipelineEncoder(model_parameters["pipeline_encoder"])

    def forward(self, heterogen_pipelines: Sequence[HeterogeneousBatch]) -> Tensor:
        raise NotImplementedError("The method should be overriden.")

    def training_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        heterogen_pipelines, y = batch
        scores = self.forward(heterogen_pipelines)
        if self.loss_name == "kl_div":
            loss = F.kl_div(
                torch.log_softmax(scores, dim=1),
                torch.log_softmax(y, dim=1),
                log_target=True,
            )
        else:
            loss = self.loss_fn(scores, y)
        self.log("train_loss", loss)
        return loss

    def evaluation_step(
        self,
        prefix: str,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
    ):
        heterogen_pipelines, y = batch
        with torch.no_grad():
            scores = self.forward(heterogen_pipelines)

        if "ndcg" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "ndcg")
            self.log(f"{prefix}_ndcg", metric_fn(y.cpu(), scores.cpu()))
        if "precision" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "precision")
            self.log(f"{prefix}_precision", metric_fn(y, scores))
        if "kendalltau" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "kendalltau")
            self.log(f"{prefix}_kendalltau", metric_fn(y.cpu(), scores.cpu()))

    def validation_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ):
        self.evaluation_step("val", batch)

    def test_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ):
        self.evaluation_step("test", batch)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def transfer_batch_to_device(self, batch: Sequence, device: torch.device, dataloader_idx: int) -> Sequence:
        def transfer_heterogeneous_batch(heterogen_batch: HeterogeneousBatch) -> HeterogeneousBatch:
            heterogen_batch.batch = heterogen_batch.batch.to(device)
            heterogen_batch.ptr = heterogen_batch.ptr.to(device)
            heterogen_batch.edge_index = heterogen_batch.edge_index.to(device)
            heterogen_batch.node_idxes_per_type = {
                k: v.to(device) for k, v in heterogen_batch.node_idxes_per_type.items()
            }
            heterogen_batch.hparams = {k: v.to(device) for k, v in heterogen_batch.hparams.items()}
            heterogen_batch.encoded_type = {k: v.to(device) for k, v in heterogen_batch.encoded_type.items()}
            return heterogen_batch

        res = []
        for e in batch:
            if isinstance(e, Tensor):
                res.append(e.to(device))
            elif isinstance(e, HeterogeneousBatch):
                e = transfer_heterogeneous_batch(e)
                res.append(e)
            elif isinstance(e, Iterable) and isinstance(e[0], HeterogeneousBatch):
                res.append([transfer_heterogeneous_batch(h) for h in e])
            else:
                raise TypeError(f"Uknown type f{type(e)}")
        return res
