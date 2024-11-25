from typing import Any, Dict

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch

from automl_surrogate.data import HeterogeneousBatch
from automl_surrogate.layers.encoders import GraphTransformer, SimpleGNNEncoder
from automl_surrogate.models.node_homogenizer import build_node_homogenizer


class PipelineEncoder(nn.Module):
    def __init__(self, model_parameters: Dict[str, Any]):
        super().__init__()
        self.node_homogenizer = build_node_homogenizer(model_parameters["node_homogenizer"])
        self.gnn = self.build_gnn(
            self.node_homogenizer.out_dim,
            model_parameters["gnn"],
        )
        self.out_dim = self.gnn.out_dim

    def homogenize(self, heterogen_pipeline: HeterogeneousBatch) -> Batch:
        homogen_nodes = self.node_homogenizer(heterogen_pipeline)
        return heterogen_pipeline.to_pyg_batch(self.node_homogenizer.out_dim, homogen_nodes)

    def forward(self, heterogen_pipeline: HeterogeneousBatch) -> Tensor:
        homogen_pipelines = self.homogenize(heterogen_pipeline)
        pipelines_embedding = self.gnn(homogen_pipelines)
        return pipelines_embedding

    @staticmethod
    def build_gnn(in_dim: int, config: Dict[str, Any]) -> nn.Module:
        config["in_size"] = in_dim
        if config["type"] == "simple_graph_encoder":
            return SimpleGNNEncoder(**{k: v for k, v in config.items() if k != "type"})
        elif config["type"] == "graph_transformer":
            return GraphTransformer(**{k: v for k, v in config.items() if k != "type"})
