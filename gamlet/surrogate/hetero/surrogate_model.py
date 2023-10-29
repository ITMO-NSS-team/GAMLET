from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Batch, Data
from torchvision.ops import MLP

from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.data_preparation.surrogate_dataset.hetero import HeterogeneousBatch, HeterogeneousData
from meta_automl.surrogate.encoders import ColumnDatasetEncoder, GraphTransformer, MLPDatasetEncoder, SimpleGNNEncoder
from meta_automl.surrogate.hetero.deterministic_graph_embedding import deterministic_graph_embedding
from meta_automl.surrogate.hetero.node_embedder import build_node_embedder
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel, RankingPipelineSurrogateModel


class HeteroPipelineDatasetRankingSurrogateModel(RankingPipelineDatasetSurrogateModel):
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        *args,
        **kwargs,
    ):
        super().__init__(model_parameters, *args, **kwargs)
        self.pipeline_extractor = FEDOTPipelineFeaturesExtractor(**model_parameters["pipeline_extractor"])
        self.node_embedder = build_node_embedder(model_parameters["node_embedder"])

    def training_step(
        self, batch: Tuple[HeterogeneousBatch, HeterogeneousBatch, Batch, torch.Tensor], *args, **kwargs
    ) -> Tensor:
        pipe1, pipe2, dset_data, y = batch
        node_embedding1 = self.node_embedder.op_hyperparams_embedder(pipe1.x)
        node_embedding2 = self.node_embedder.op_hyperparams_embedder(pipe2.x)
        x_pipe1 = pipe1.to_pyg_batch(self.node_embedder.op_hyperparams_embedder.out_dim, node_embedding1)
        x_pipe2 = pipe2.to_pyg_batch(self.node_embedder.op_hyperparams_embedder.out_dim, node_embedding2)

        return super().training_step(
            (x_pipe1, x_pipe2, dset_data, y), *args, **kwargs
        )  # Why dataset features are of Data type, not Tensor?

    def validation_step(self, batch: Tuple[Tensor, Tensor, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe, x_dset, y_true = batch
        node_embedding = self.node_embedder.op_hyperparams_embedder(pipe.x)
        x_pipe = pipe.to_pyg_batch(self.node_embedder.op_hyperparams_embedder.out_dim, node_embedding)
        return super().validation_step(
            (task_id, pipe_id, x_pipe, x_dset, y_true), *args, **kwargs
        )  # Why dataset features are of Data type, not Tensor?

    def test_step(self, batch: Tuple[Tensor, Tensor, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe, x_dset, y_true = batch
        node_embedding = self.node_embedder.op_hyperparams_embedder(pipe.x)
        x_pipe = pipe.to_pyg_batch(self.node_embedder.op_hyperparams_embedder.out_dim, node_embedding)
        return super().test_step(
            (task_id, pipe_id, x_pipe, x_dset, y_true), *args, **kwargs
        )  # Why dataset features are of Data type, not Tensor?


class HeteroPipelineRankingSurrogateModel(RankingPipelineSurrogateModel):
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        *args,
        **kwargs,
    ):
        super().__init__(model_parameters, *args, **kwargs)
        self.pipeline_extractor = FEDOTPipelineFeaturesExtractor(**model_parameters["pipeline_extractor"])
        self.node_embedder = build_node_embedder(model_parameters["node_embedder"])

    def training_step(
        self, batch: Tuple[HeterogeneousBatch, HeterogeneousBatch, torch.Tensor], *args, **kwargs
    ) -> Tensor:
        pipe1, pipe2, y = batch
        node_embedding1 = self.node_embedder(pipe1)
        node_embedding2 = self.node_embedder(pipe2)
        x_pipe1 = pipe1.to_pyg_batch(self.node_embedder.out_dim, node_embedding1)
        x_pipe2 = pipe2.to_pyg_batch(self.node_embedder.out_dim, node_embedding2)
        return super().training_step((x_pipe1, x_pipe2, y), *args, **kwargs)

    def validation_step(self, batch: Tuple[Tensor, Tensor, HeterogeneousBatch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe, y_true = batch
        node_embedding = self.node_embedder(pipe)
        x_pipe = pipe.to_pyg_batch(self.node_embedder.out_dim, node_embedding)
        return super().validation_step((task_id, pipe_id, x_pipe, y_true), *args, **kwargs)

    def test_step(self, batch: Tuple[Tensor, Tensor, HeterogeneousBatch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe, y_true = batch
        node_embedding = self.node_embedder(pipe)
        x_pipe = pipe.to_pyg_batch(self.node_embedder.out_dim, node_embedding)
        return super().test_step((task_id, pipe_id, x_pipe, y_true), *args, **kwargs)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        res = []
        for e in batch:
            if isinstance(e, Tensor):
                res.append(e.to(device))
            elif isinstance(e, HeterogeneousBatch):
                e.batch = e.batch.to(device)
                e.edge_index = e.edge_index.to(device)
                e.node_idxes_per_type = {k: v.to(device) for k, v in e.node_idxes_per_type.items()}
                e.hparams = {k: v.to(device) for k, v in e.hparams.items()}
                e.encoded_type = {k: v.to(device) for k, v in e.encoded_type.items()}
                res.append(e)
            else:
                raise TypeError(f"Uknown type f{type(e)}")
        return res

# # Model that concats embedding of two graphs and then produce logit
# class HeteroPipelineRankingSurrogateModel(RankingPipelineSurrogateModel):
#     def __init__(
#         self,
#         model_parameters: Dict[str, Any],
#         lr: float = 1e-3,
#         weight_decay: float = 1e-4,
#         temperature: float = 10,
#     ):
#         LightningModule.__init__(self)

#         if model_parameters["pipeline_encoder"]["type"] == "simple_graph_encoder":
#             config = model_parameters["pipeline_encoder"]
#             self.pipeline_encoder = SimpleGNNEncoder(**{k: v for k, v in config.items() if k != "type"})
#         elif model_parameters["pipeline_encoder"]["type"] == "graph_transformer":
#             config = model_parameters["pipeline_encoder"]
#             self.pipeline_encoder = GraphTransformer(**{k: v for k, v in config.items() if k != "type"})

#         self.final_model = nn.Sequential(
#             nn.BatchNorm1d(self.pipeline_encoder.out_dim * 2),
#             nn.Linear(self.pipeline_encoder.out_dim * 2, self.pipeline_encoder.out_dim),
#             nn.ReLU(),
#             nn.Linear(self.pipeline_encoder.out_dim, self.pipeline_encoder.out_dim),
#             nn.ReLU(),
#             nn.Linear(self.pipeline_encoder.out_dim, 1),
#         )

#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.temperature = temperature

#         # Migration to pytorch_lightning > 1.9.5
#         self.validation_step_outputs = []
#         self.test_step_outputs = []
#         self.save_hyperparameters()  # TODO: is it required? We have config file.

#         self.K_TOP = 3
#         self.pipeline_extractor = FEDOTPipelineFeaturesExtractor(**model_parameters["pipeline_extractor"])
#         self.node_embedder = build_node_embedder(model_parameters["node_embedder"])

#     def training_step(
#         self, batch: Tuple[HeterogeneousBatch, HeterogeneousBatch, torch.Tensor], *args, **kwargs
#     ) -> Tensor:
#         pipe1, pipe2, y = batch
#         node_embedding1 = self.node_embedder(pipe1)
#         node_embedding2 = self.node_embedder(pipe2)
#         x_pipe1 = pipe1.to_pyg_batch(self.node_embedder.out_dim, node_embedding1)
#         x_pipe2 = pipe2.to_pyg_batch(self.node_embedder.out_dim, node_embedding2)

#         z_pipeline1 = self.pipeline_encoder(x_pipe1)
#         z_pipeline2 = self.pipeline_encoder(x_pipe2)

#         z_pipeline = torch.hstack([z_pipeline1, z_pipeline2])
#         logit = self.final_model(z_pipeline)
#         logit = torch.squeeze(logit)
#         loss = nn.functional.binary_cross_entropy_with_logits(logit * self.temperature, y)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch: Tuple[Tensor, Tensor, HeterogeneousBatch, Tensor], *args, **kwargs: Any) -> None:
#         pass

#     def test_step(self, batch: Tuple[Tensor, Tensor, HeterogeneousBatch, Tensor], *args, **kwargs: Any) -> None:
#         pass

#     def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
#         res = []
#         for e in batch:
#             if isinstance(e, Tensor):
#                 res.append(e.to(device))
#             elif isinstance(e, HeterogeneousBatch):
#                 e.batch = e.batch.to(device)
#                 e.edge_index = e.edge_index.to(device)
#                 e.node_idxes_per_type = {k: v.to(device) for k, v in e.node_idxes_per_type.items()}
#                 e.hparams = {k: v.to(device) for k, v in e.hparams.items()}
#                 e.encoded_type = {k: v.to(device) for k, v in e.encoded_type.items()}
#                 res.append(e)
#             else:
#                 raise TypeError(f"Uknown type f{type(e)}")
#         return res

#     def on_validation_epoch_end(self) -> None:
#         self.log("val_ndcg", -1)

#     def on_test_epoch_end(self) -> None:
#         self.log("val_ndcg", -1)

# ---------------------------------------

# class HeteroPipelineDatasetRankingSurrogateModelDetermenisticEmbedding(RankingPipelineDatasetSurrogateModel):
#     def __init__(
#         self,
#         model_parameters: Dict[str, Any],
#         lr: float = 1e-3,
#         weight_decay: float = 1e-4,
#         temperature: float = 10,
#     ):
#         LightningModule.__init__(self)

#         self.pipeline_encoder = MLP(
#             in_channels=model_parameters["pipeline_encoder"]["in_size"],
#             hidden_channels=[model_parameters["pipeline_encoder"]["d_model"]]
#             * model_parameters["pipeline_encoder"]["num_layers"],
#             dropout=model_parameters["pipeline_encoder"]["dropout"],
#             norm_layer=nn.BatchNorm1d if model_parameters["pipeline_encoder"]["batch_norm"] else nn.LayerNorm,
#             inplace=False,
#         )
#         self.pipeline_encoder.out_dim = model_parameters["pipeline_encoder"]["d_model"]

#         if model_parameters["dataset_encoder"]["type"] == "column":
#             config = model_parameters["dataset_encoder"]
#             self.dataset_encoder = ColumnDatasetEncoder(
#                 input_dim=config["dim_dataset"],
#                 hidden_dim=config["d_model_dset"],
#                 output_dim=config["d_model_dset"],
#             )
#         elif model_parameters["dataset_encoder"]["type"] == "aggregated":
#             config = model_parameters["dataset_encoder"]
#             self.dataset_encoder = MLPDatasetEncoder(
#                 input_dim=config["dim_dataset"],
#                 hidden_dim=config["d_model_dset"],
#                 output_dim=config["d_model_dset"],
#             )
#         else:
#             raise ValueError("dataset_encoder_type should be 'column' or 'aggregated'")

#         cat_dim = self.dataset_encoder.out_dim + self.pipeline_encoder.out_dim
#         self.final_model = nn.Sequential(
#             nn.BatchNorm1d(cat_dim),
#             nn.Linear(cat_dim, cat_dim),
#             nn.ReLU(),
#             nn.Linear(cat_dim, 1),
#         )

#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.temperature = temperature

#         # Migration to pytorch_lightning > 1.9.5
#         self.validation_step_outputs = []
#         self.test_step_outputs = []
#         self.save_hyperparameters()  # TODO: is it required? We have config file.

#         self.K_TOP = 3

#     def training_step(self, batch: Tuple[List[str], List[str], Batch, torch.Tensor], *args, **kwargs) -> Tensor:
#         pipe1_json_str, pipe2_json_str, dset_data, y = batch

#         x_pipe1 = torch.FloatTensor(np.vstack([deterministic_graph_embedding(p) for p in pipe1_json_str])).to(
#             self.device
#         )
#         x_pipe2 = torch.FloatTensor(np.vstack([deterministic_graph_embedding(p) for p in pipe2_json_str])).to(
#             self.device
#         )
#         return super().training_step(
#             (x_pipe1, x_pipe2, dset_data, y), *args, **kwargs
#         )  # Why dataset features are of Data type, not Tensor?

#     def validation_step(self, batch: Tuple[Tensor, Tensor, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
#         task_id, pipe_id, pipe_json_str, x_dset, y_true = batch
#         x_graph = torch.FloatTensor(np.vstack([deterministic_graph_embedding(p) for p in pipe_json_str])).to(
#             self.device
#         )
#         return super().validation_step(
#             (task_id, pipe_id, x_graph, x_dset, y_true), *args, **kwargs
#         )  # Why dataset features are of Data type, not Tensor?

#     def test_step(self, batch: Tuple[Tensor, Tensor, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
#         task_id, pipe_id, pipe_json_str, x_dset, y_true = batch
#         x_graph = torch.FloatTensor(np.vstack([deterministic_graph_embedding(p) for p in pipe_json_str])).to(
#             self.device
#         )
#         return super().test_step(
#             (task_id, pipe_id, x_graph, x_dset, y_true), *args, **kwargs
#         )  # Why dataset features are of Data type, not Tensor?
