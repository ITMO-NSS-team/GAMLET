from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Batch, Data
from torchvision.ops import MLP

from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.surrogate.encoders import ColumnDatasetEncoder, GraphTransformer, MLPDatasetEncoder, SimpleGNNEncoder
from meta_automl.surrogate.hetero.deterministic_graph_embedding import deterministic_graph_embedding
from meta_automl.surrogate.hetero.node_embedder import build_node_embedder
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel


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

    def _pipeline_json_string2data(self, pipeline_json_str: str) -> Data:
        nodes = self.pipeline_extractor._get_nodes_from_json_string(pipeline_json_str)
        # Add artificial `dataset` node to make minimal graph length > 1 to avoid errors in pytorch_geometric.
        nodes = self.pipeline_extractor._append_dataset_node(nodes)
        edge_index = self.pipeline_extractor._get_edge_index_tensor(nodes)
        operations_ids = self.pipeline_extractor._get_operations_ids(nodes)
        operations_names = self.pipeline_extractor._get_operations_names(nodes, operations_ids)
        operations_parameters = self.pipeline_extractor._get_operations_parameters(nodes, operations_ids)

        node_embeddings: List[Tensor] = []
        for op_name, op_params in zip(operations_names, operations_parameters):
            name_vec = self.pipeline_extractor._operation_name2vec(op_name)
            name_tensor = torch.FloatTensor(name_vec.reshape(1, -1)).to(self.device)
            parameters_vector = self.pipeline_extractor._operation_parameters2vec(op_name, op_params)
            # TODO: check if exists already.
            parameters_tensor = torch.FloatTensor(parameters_vector.reshape(1, -1)).to(self.device)
            node_embedding = self.node_embedder(op_name, name_tensor, parameters_tensor)
            node_embeddings.append(node_embedding)

        data = Data(x=torch.vstack(node_embeddings), edge_index=edge_index, in_size=self.node_embedder.out_dim)
        return data

    def training_step(self, batch: Tuple[List[str], List[str], Batch, torch.Tensor], *args, **kwargs) -> Tensor:
        pipe1_json_str, pipe2_json_str, dset_data, y = batch
        x_pipe1 = Batch.from_data_list([self._pipeline_json_string2data(p) for p in pipe1_json_str]).to(self.device)
        x_pipe2 = Batch.from_data_list([self._pipeline_json_string2data(p) for p in pipe2_json_str]).to(self.device)
        return super().training_step(
            (x_pipe1, x_pipe2, dset_data, y), *args, **kwargs
        )  # Why dataset features are of Data type, not Tensor?

    def validation_step(self, batch: Tuple[Tensor, Tensor, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe_json_str, x_dset, y_true = batch
        x_graph = Batch.from_data_list([self._pipeline_json_string2data(p) for p in pipe_json_str])
        return super().validation_step(
            (task_id, pipe_id, x_graph, x_dset, y_true), *args, **kwargs
        )  # Why dataset features are of Data type, not Tensor?

    def test_step(self, batch: Tuple[Tensor, Tensor, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe_json_str, x_dset, y_true = batch
        x_graph = Batch.from_data_list([self._pipeline_json_string2data(p) for p in pipe_json_str])
        return super().test_step(
            (task_id, pipe_id, x_graph, x_dset, y_true), *args, **kwargs
        )  # Why dataset features are of Data type, not Tensor?


class HeteroPipelineDatasetRankingSurrogateModelDetermenisticEmbedding(RankingPipelineDatasetSurrogateModel):
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 10,
    ):
        LightningModule.__init__(self)

        self.pipeline_encoder = MLP(
            in_channels=model_parameters["pipeline_encoder"]["in_size"],
            hidden_channels=[model_parameters["pipeline_encoder"]["d_model"]]
            * model_parameters["pipeline_encoder"]["num_layers"],
            dropout=model_parameters["pipeline_encoder"]["dropout"],
            norm_layer=nn.BatchNorm1d if model_parameters["pipeline_encoder"]["batch_norm"] else nn.LayerNorm,
            inplace=False,
        )
        self.pipeline_encoder.out_dim = model_parameters["pipeline_encoder"]["d_model"]

        if model_parameters["dataset_encoder"]["type"] == "column":
            config = model_parameters["dataset_encoder"]
            self.dataset_encoder = ColumnDatasetEncoder(
                input_dim=config["dim_dataset"],
                hidden_dim=config["d_model_dset"],
                output_dim=config["d_model_dset"],
            )
        elif model_parameters["dataset_encoder"]["type"] == "aggregated":
            config = model_parameters["dataset_encoder"]
            self.dataset_encoder = MLPDatasetEncoder(
                input_dim=config["dim_dataset"],
                hidden_dim=config["d_model_dset"],
                output_dim=config["d_model_dset"],
            )
        else:
            raise ValueError("dataset_encoder_type should be 'column' or 'aggregated'")

        cat_dim = self.dataset_encoder.out_dim + self.pipeline_encoder.out_dim
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

    def training_step(self, batch: Tuple[List[str], List[str], Batch, torch.Tensor], *args, **kwargs) -> Tensor:
        pipe1_json_str, pipe2_json_str, dset_data, y = batch

        x_pipe1 = torch.FloatTensor(np.vstack([deterministic_graph_embedding(p) for p in pipe1_json_str])).to(
            self.device
        )
        x_pipe2 = torch.FloatTensor(np.vstack([deterministic_graph_embedding(p) for p in pipe2_json_str])).to(
            self.device
        )
        return super().training_step(
            (x_pipe1, x_pipe2, dset_data, y), *args, **kwargs
        )  # Why dataset features are of Data type, not Tensor?

    def validation_step(self, batch: Tuple[Tensor, Tensor, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe_json_str, x_dset, y_true = batch
        x_graph = torch.FloatTensor(np.vstack([deterministic_graph_embedding(p) for p in pipe_json_str])).to(
            self.device
        )
        return super().validation_step(
            (task_id, pipe_id, x_graph, x_dset, y_true), *args, **kwargs
        )  # Why dataset features are of Data type, not Tensor?

    def test_step(self, batch: Tuple[Tensor, Tensor, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe_json_str, x_dset, y_true = batch
        x_graph = torch.FloatTensor(np.vstack([deterministic_graph_embedding(p) for p in pipe_json_str])).to(
            self.device
        )
        return super().test_step(
            (task_id, pipe_id, x_graph, x_dset, y_true), *args, **kwargs
        )  # Why dataset features are of Data type, not Tensor?
