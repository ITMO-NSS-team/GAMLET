import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from torch_geometric.data import Data

from meta_automl.data_preparation.surrogate_dataset.hetero import HeterogeneousData


class FEDOTPipelineFeaturesExtractor:
    """FEDOT pipeline features extractor.

    List of extracted features: directed adjacency matrix, nodes operation type, nodes hyperparameters.

    Parameters:
    -----------
    operation_encoding: Type of operation encoding. Can be `"ordinal"`, `"onehot"` or `None`. Default: `"ordinal"`.
    hyperparameters_embedder: Callable object to embed hyperparameters. Default: `None`.
                              The object should accept an operation name and the operation
                              hyperparameters vector of shape `[1, N]`.
    """

    def __init__(
        self,
        operation_encoding: Optional[str] = "ordinal",
        hyperparameters_embedder: Optional[Callable] = None,
    ):
        err_msg = "At least one of `operation_encoding` and `hyperparameters_embedder` must be provided."
        if operation_encoding is None and hyperparameters_embedder is None:
            raise ValueError(err_msg)

        models_repo = OperationTypesRepository()
        self.operation_types = []
        for k in models_repo.__initialized_repositories__.keys():
            for o in models_repo.__initialized_repositories__[k]:
                self.operation_types.append(o.id)
        self.operation_types += ["dataset"]  # Artificial node that is appended to pipeline. Represents data source.
        self.operations_count = len(self.operation_types)
        self.operations_space = PipelineSearchSpace().get_parameters_dict()

        possible_operation_encodings = ["ordinal", "onehot", None]
        assert_message = f"Expected `return_type` is of {possible_operation_encodings}, got {operation_encoding}"
        assert operation_encoding in possible_operation_encodings, assert_message

        self.hyperparameters_embedder = hyperparameters_embedder
        self.operation_encoding = operation_encoding
        if operation_encoding is not None:
            self.operation_name2vec = self._get_operation_name2vec()

    def _get_operation_name2vec(self) -> Dict[str, Union[int, np.ndarray]]:
        result = {}
        for i, operation_name in enumerate(self.operation_types):
            if self.operation_encoding == "onehot":
                vector = np.zeros(self.operations_count)
                vector[i] = 1
                result[operation_name] = vector
            elif self.operation_encoding == "ordinal":
                vector = np.array(i).reshape(
                    1,
                )
                result[operation_name] = vector
            else:
                raise ValueError(f"Unsuppored operation encoding: {self.operation_encoding}")
        return result

    def _get_nodes_from_json_string(self, json_string: str) -> List[Dict[str, Any]]:
        data = json.loads(json_string)
        nodes = data["nodes"]
        return nodes

    def _get_operations_ids(self, nodes: List[Dict[str, Any]]) -> List[int]:
        return [node["operation_id"] for node in nodes]

    def _get_operations_names(self, nodes: List[Dict[str, Any]], order: List[int] = None) -> List[str]:
        operations_names = []
        if order is None:
            for node in nodes:
                operation_name = node["operation_name"]
                if operation_name is None:  # TODO: is it a workaround or normal solution?
                    operation_type = node["operation_type"]
                operations_names.append(operation_type)
        else:
            for index in order:
                operation_name = nodes[index]["operation_name"]
                if operation_name is None:
                    operation_type = nodes[index]["operation_type"]
                operations_names.append(operation_type)
        return operations_names

    def _operation_name2vec(self, operation_name: str) -> np.ndarray:
        return self.operation_name2vec[operation_name]

    def _operation_parameters2vec(self, operation_name: str, operation_parameters: Dict[str, Any]) -> np.ndarray:
        if operation_name not in self.operations_space:
            hyperparams_vec = np.asarray([])
        else:
            space = self.operations_space[operation_name]
            hyperparams_vec = []
            for parameter_name in space:  # Iterate over space keys to keep number of parameters and their order.
                parameter_type = space[parameter_name]["type"]
                assert_msg = f"Unsupported parameter type: {parameter_type}"
                assert parameter_type in ["categorical", "continuous", "discrete"], assert_msg

                if parameter_name in operation_parameters:
                    parameter_value = operation_parameters[parameter_name]
                    if parameter_type == "categorical":  # Do Ordinal encoding.
                        parameter_value = space[parameter_name]["sampling-scope"][0].index(parameter_value)
                    else:  # Do Min-Max scaling.
                        lower_bound = space[parameter_name]["sampling-scope"][0]
                        upper_bound = space[parameter_name]["sampling-scope"][1]
                        parameter_value = (parameter_value - lower_bound) / (upper_bound - lower_bound)
                else:
                    parameter_value = -1
                hyperparams_vec.append(parameter_value)
        hyperparams_vec = np.asarray(hyperparams_vec)
        return hyperparams_vec

    def _operation2vec(self, operation_name: str, operation_parameters: Dict[str, Any]) -> np.ndarray:
        name_vec = np.asarray([])
        if self.operation_encoding is not None:
            name_vec = self._operation_name2vec(operation_name)
        parameters_vec = np.asarray([])
        if self.hyperparameters_embedder:
            parameters_vec = self._operation_parameters2vec(operation_name, operation_parameters)
            parameters_vec = self.hyperparameters_embedder(operation_name, parameters_vec.reshape(1, -1)).reshape(-1)
        return np.hstack((name_vec, parameters_vec))

    def _operations2tensor(
        self,
        operations_names: List[str],
        operations_parameters: List[Dict[str, Any]],
    ) -> torch.Tensor:
        tensor = np.vstack([self._operation2vec(n, p) for n, p in zip(operations_names, operations_parameters)])
        return torch.Tensor(tensor).to(dtype=torch.float32)

    def _get_operations_parameters(self, nodes: List[Dict[str, Any]], order: List[int] = None) -> List[Dict[str, Any]]:
        def extract_parameters(node: Dict[str, Any]) -> Dict[str, Any]:
            node_parameters = {}
            node_parameters.update(node["params"])
            node_parameters.update(node["custom_params"])
            return node_parameters

        if order is None:
            return [extract_parameters(node) for node in nodes]
        else:
            return [extract_parameters(nodes[index]) for index in order]

    def _get_edge_index_tensor(self, nodes: List[Dict[str, Any]]) -> torch.LongTensor:
        edges = []

        for node in nodes:
            nodes_from = node["nodes_from"]
            if len(nodes_from) > 0:
                target = node["operation_id"]
                for source in nodes_from:
                    edges.append([source, target])

        return torch.LongTensor(edges).T

    @staticmethod
    def _append_dataset_node(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_op_id = max([node["operation_id"] for node in nodes]) + 1
        for node in nodes:
            if not node["nodes_from"]:
                node["nodes_from"] = [max_op_id]
        dataset_node = {
            "operation_id": max_op_id,
            "operation_type": "dataset",
            "operation_name": None,
            "custom_params": {},
            "params": {},
            "nodes_from": [],
        }
        dataset_node = [
            dataset_node,
        ]
        nodes = dataset_node + nodes
        return nodes

    def _get_data(self, pipeline_json_string: str) -> Data:
        nodes = self._get_nodes_from_json_string(pipeline_json_string)
        # Add artificial `dataset` node to make minimal graph length > 1 to avoid errors in pytorch_geometric.
        nodes = self._append_dataset_node(nodes)

        edge_index = self._get_edge_index_tensor(nodes)

        operations_ids = self._get_operations_ids(nodes)
        operations_names = self._get_operations_names(nodes, operations_ids)
        operations_parameters = self._get_operations_parameters(nodes, operations_ids)
        operations_tensor = self._operations2tensor(operations_names, operations_parameters)

        data = Data(x=operations_tensor, edge_index=edge_index, in_size=operations_tensor.shape[1])
        return data

    def __call__(self, pipeline_json_string: str):
        return self._get_data(pipeline_json_string)


class FEDOTPipelineFeaturesExtractor2(FEDOTPipelineFeaturesExtractor):
    def __init__(
        self,
        operation_encoding: Optional[str] = "ordinal",
    ):
        super().__init__(operation_encoding, object())

    def _operation2tensor(self, operation_name: str, operation_parameters: Dict[str, Any]) -> torch.Tensor:
        op_vec = self._operation2vec(operation_name, operation_parameters)
        return torch.FloatTensor(op_vec.reshape(1, -1))

    def _operation2vec(self, operation_name: str, operation_parameters: Dict[str, Any]) -> np.ndarray:
        name_vec = np.asarray([])
        if self.operation_encoding is not None:
            name_vec = self._operation_name2vec(operation_name)
        parameters_vec = self._operation_parameters2vec(operation_name, operation_parameters)
        return np.hstack((name_vec, parameters_vec))

    def _get_data(self, pipeline_json_string: str) -> HeterogeneousData:
        nodes = self._get_nodes_from_json_string(pipeline_json_string)
        # Add artificial `dataset` node to make minimal graph length > 1 to avoid errors in pytorch_geometric.
        nodes = self._append_dataset_node(nodes)

        edge_index = self._get_edge_index_tensor(nodes)

        operations_ids = self._get_operations_ids(nodes)
        operations_names = self._get_operations_names(nodes, operations_ids)
        operations_parameters = self._get_operations_parameters(nodes, operations_ids)

        node_idxes_per_type = defaultdict(list)
        for node_type, node_index in zip(operations_names, operations_ids):
            node_idxes_per_type[node_type].append(node_index)

        hparams = defaultdict(list)
        encoded_type = defaultdict(list)
        for operation_name, operation_parameters in zip(operations_names, operations_parameters):
            parameters_vec = self._operation_parameters2vec(operation_name, operation_parameters)
            parameters_vec = torch.FloatTensor(parameters_vec.reshape(1, -1))
            hparams[operation_name].append(parameters_vec)
            if self.operation_encoding is not None:
                name_vec = self._operation_name2vec(operation_name)
                name_vec = torch.LongTensor(name_vec.reshape(1, -1))
                encoded_type[operation_name].append(name_vec)

        for operation_name, operation_parameters in hparams.items():
            hparams[operation_name] = torch.vstack(operation_parameters)

        if self.operation_encoding is not None:
            for operation_name, operation_encoding in encoded_type.items():
                encoded_type[operation_name] = torch.vstack(operation_encoding)
        else:
            encoded_type = {}

        data = HeterogeneousData(
            edge_index=edge_index,
            node_idxes_per_type=node_idxes_per_type,
            hparams=hparams,
            encoded_type=encoded_type,
        )
        return data
