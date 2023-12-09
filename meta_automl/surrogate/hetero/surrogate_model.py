from typing import Any, Dict, List, Tuple

import torch
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.surrogate.hetero.node_embedder import build_node_embedder
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel
from torch import Tensor
from torch_geometric.data import Batch, Data


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

    def training_step(self, batch: Tuple[List[str], List[str], torch.Tensor, torch.Tensor], *args, **kwargs) -> Tensor:
        pipe1_json_str, pipe2_json_str, dset_data, y = batch
        x_pipe1 = Batch.from_data_list([self._pipeline_json_string2data(p) for p in pipe1_json_str])
        x_pipe2 = Batch.from_data_list([self._pipeline_json_string2data(p) for p in pipe2_json_str])
        return super().training_step((x_pipe1, x_pipe2, Data(x=dset_data), y), *args, **kwargs)  # Why dataset features are of Data type, not Tensor?

    def validation_step(self, batch: Tuple[Tensor, Batch, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe_json_str, x_dset, y_true = batch
        x_graph = Batch.from_data_list([self._pipeline_json_string2data(p) for p in pipe_json_str])
        return super().validation_step((task_id, pipe_id, x_graph, Data(x=x_dset), y_true), *args, **kwargs)  # Why dataset features are of Data type, not Tensor?

    def test_step(self, batch: Tuple[Tensor, Batch, List[str], Batch, Tensor], *args, **kwargs: Any) -> None:
        task_id, pipe_id, pipe_json_str, x_dset, y_true = batch
        x_graph = Batch.from_data_list([self._pipeline_json_string2data(p) for p in pipe_json_str])
        return super().test_step((task_id, pipe_id, x_graph, Data(x=x_dset), y_true), *args, **kwargs)  # Why dataset features are of Data type, not Tensor?
