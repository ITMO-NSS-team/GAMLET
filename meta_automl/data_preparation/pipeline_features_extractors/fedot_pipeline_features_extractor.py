import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data.data import Data


class FEDOTPipelineFeaturesExtractor:
    """FEDOT pipeline features extractor.

    List of extracted features: directed adjacency matrix, nodes operation type.

    Parameters:
    -----------
    include_operations_hyperparameters: Whether to include operation hyperparameters. Currently not supported. Default: `False`.
    operation_type2name: Mapping of operation type to its implementation name. Default mapping is adapted from FEDOT.
    operation_encoding: Type of operation encoding. Can be `"ordinal"` or `"onehot"`. Default: `"ordinal"`.
    """

    default_operation_type2name = {
        '': '',  # Empty string indicates that the node means <no_operation> as https://t.me/FGksjp67 requested.
        'scaling': 'ScalingImplementation',
        'normalization': 'NormalizationImplementation',
        'pca': 'PCAImplementation',
        'lgbm': 'LGBMClassifier',
        'mlp': 'MLPClassifier',
        'bernb': 'BernoulliNB',
        'isolation_forest_class': 'IsolationForestClassImplementation',
        'fast_ica': 'FastICAImplementation',
        'rf': 'RandomForestClassifier',
        'dt': 'DecisionTreeClassifier',
        'qda': 'QDAImplementation',
        'knn': 'FedotKnnClassImplementation',
        'resample': 'ResampleImplementation',
        'logit': 'LogisticRegression',
        'poly_features': 'PolyFeaturesImplementation'
    }

    def __init__(
            self,
            include_operations_hyperparameters: Optional[bool] = False,
            operation_type2name: Optional[Dict[str, str]] = default_operation_type2name,
            operation_encoding: Optional[str] = "ordinal",
        ):
        possible_operation_encodings = ["ordinal", "onehot"]
        assert_message = f"Expected `return_type` is of {possible_operation_encodings}, got {operation_encoding}"
        assert operation_encoding in possible_operation_encodings, assert_message

        self.include_operations_hyperparameters = include_operations_hyperparameters
        self.operation_type2name = operation_type2name
        self.operation_encoding = operation_encoding
        self.operation_name2vec = self._get_operation_name2vec()

    def _get_operation_name2vec(self) -> Dict[str, Union[int, np.ndarray]]:
        result = {}
        for i, operation_name in enumerate(self.operation_type2name.values()):
            if self.operation_encoding == "onehot":
                vector = np.zeros(len(self.operation_type2name))
                vector[i] = 1
                result[operation_name] = vector
            elif self.operation_encoding == "ordinal":
                vector = np.array(i).reshape(1,)
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
                    operation_name = self.operation_type2name[operation_type]
                operations_names.append(operation_name)
        else:
            for index in order:
                operation_name = nodes[index]["operation_name"]
                if operation_name is None:
                    operation_type = nodes[index]["operation_type"]
                    operation_name = self.operation_type2name[operation_type]
                operations_names.append(operation_name)
        return operations_names

    def _operation_name2vec(self, operation_name: str) -> np.ndarray:
        return self.operation_name2vec[operation_name]

    def _parameters2vec(self, operation_name: str, operation_parameters: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError("Currently, operation features are not supported")

    def _operation2vec(self, operation_name: str, operation_parameters: Dict[str, Any]) -> np.ndarray:
        name_vec = self._operation_name2vec(operation_name)
        if self.include_operations_hyperparameters:
            parameters_vec = self._parameters2vec(operation_name, operation_parameters)
            return np.hstack((name_vec, parameters_vec))
        else:
            return name_vec

    def _operations2tensor(
            self,
            operations_names: List[str],
            operations_parameters: List[Dict[str, Any]],
    ) -> torch.Tensor:
        tensor = np.vstack([self._operation2vec(n, p) for n, p in zip(operations_names, operations_parameters)])
        return torch.Tensor(tensor)

    def _get_operations_parameters(self, nodes: List[Dict[str, Any]], order: List[int] = None) -> List[Dict[str, Any]]:
        if order is None:
            return [node["params"] for node in nodes]
        else:
            return [nodes[index]["params"] for index in order]

    def _get_edge_index_tensor(self, nodes: List[Dict[str, Any]]) -> torch.LongTensor:
        edges = []
        for node in nodes:
            nodes_from = node["nodes_from"]
            if len(nodes_from) > 0:
                target = node["operation_id"]
                for source in nodes_from:
                    edges.append([source, target])
        return torch.LongTensor(edges).T

    def _get_data(self, pipeline_json_string: str) -> Data:
        nodes = self._get_nodes_from_json_string(pipeline_json_string)
        operations_ids = self._get_operations_ids(nodes)
        operations_names = self._get_operations_names(nodes, operations_ids)
        operations_parameters = self._get_operations_parameters(nodes, operations_ids)
        operations_tensor = self._operations2tensor(operations_names, operations_parameters)
        edge_index = self._get_edge_index_tensor(nodes)
        data = Data(operations_tensor, edge_index)
        return data


    def __call__(self, pipeline_json_string: str):
        return self._get_data(pipeline_json_string)