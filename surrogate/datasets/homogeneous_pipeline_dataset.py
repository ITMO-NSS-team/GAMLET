import json
import os
import pickle
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch_geometric.data import Dataset
from torch_geometric.data.data import Data

# Can be subject of changes
type_name_mapping = {
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


# TODO: target scaling is due to biased dataset at internship. If metric is always within `[0,1]` range can be removed.
class HomogeneousPipelineDataset(Dataset):
    """
    Dataset that produce a pipeline as `pytorch_geometric.data.Data`
    those nodes properties aligned across all possible operations.

    A node consists of:
    * onehot-encoded operation type
    * list of the operation hyperparameters with `-1` denoting parameters that belong to other operations.

    Target is the pipeline `f1_score` and `roc_auc`.

    Node hyperparameters and target are scaled with MinMaxScaler.

    At the first run or if `root` directory has no `preprocessed` subdirectory,
    `OneHotEncoder` and `MinMaxScalers` are fitted.

    Args:
        root: Dataset root directory.
            Expected to have `metrics` and `pipelines` subdirectories.
            Metrics are `pickle`-files with `dict` of metrics and pipelines are `json`-files from `FEDOT`.
            A sample metrics and pipeline should have the same name (e.g. `pipeline1.json` and `pipeline1.pickle`).
        split: Name of the dataset split. Can be `train`, `val` or `test` or `None` (denotes all dataset).
            Split ratios are `0.7`, `0.15`, `0.15`, `1` accordingly.
        log: Flag whether to print any console output while downloading and processing the dataset.
        direction: Edge direction. Can be `undirected`, `directed` (as in original pipeline)
            or `reversed` (reversed original direction). Edge direction might have impact on
            node embedding aggregation.
        use_operations_hyperparameters: Flag whether to include operations hyperparameters to node features.
        overriden_processed_dir: path to processed dir to override default. # TODO: fix to run knowledge_base_v0
    """

    # Path to fitted preprocessors files that are stored in dataset `root` `processed` subdir.
    OPERATION_NAME_ONE_HOT_ENCODER_FILENAME = "operation_name_one_hot_encoder_filename.pickle"
    OPERATIONS_PREPROCESSORS_FILENAME = "operations_preprocessors_filename.pickle"
    OPERATIONS_PARAMETERS_VECTOR_INDEXES_FILENAME = "operations_parameters_vector_indexes_filename.pickle"
    OPERATIONS_PARAMETERS_VECTOR_TEMPLATE_FILENAME = "operations_parameters_vector_template_filename.pickle"
    METRICS_SCALER_FILENAME = "metrics_scaler_filename.pickle"
    # Predefined train/val/test ratio.
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15

    def __init__(
            self,
            root: str,
            split: str = None,
            log: bool = True,
            direction: str = "undirected",
            use_operations_hyperparameters: bool = True,
            overriden_processed_dir: str = None,
    ):
        self.direction = direction
        self.use_operations_hyperparameters = use_operations_hyperparameters
        self.type_name_mapping = type_name_mapping
        self.json_pipelines = list(Path(os.path.join(root, "pipelines")).glob("**/*.json"))
        self.pickle_metrics = list(Path(os.path.join(root, "metrics")).glob("**/*.pickle"))
        super().__init__(root, log=log)
        if split is not None:
            if split == "train":
                start = 0
                stop = int(self.TRAIN_RATIO * len(self.json_pipelines))
            elif split == "val":
                start = int(self.TRAIN_RATIO * len(self.json_pipelines))
                stop = int((self.TRAIN_RATIO + self.VAL_RATIO) * len(self.json_pipelines))
            elif split == "test":
                start = int((self.TRAIN_RATIO + self.VAL_RATIO) * len(self.json_pipelines))
                stop = len(self.json_pipelines)
            else:
                raise ValueError(f"Unknown split: {split}")
            self.json_pipelines = self.json_pipelines[start: stop]
            self.pickle_metrics = self.pickle_metrics[start: stop]

        # Fix to run knowledge_base_v0
        processed_dir = overriden_processed_dir if overriden_processed_dir is not None else self.processed_dir

        self._operations_parameters_vector_indexes = self._load_pickle(
            os.path.join(processed_dir, self.OPERATIONS_PARAMETERS_VECTOR_INDEXES_FILENAME),
        )
        self._operations_parameters_vector_template = self._load_pickle(
            os.path.join(processed_dir, self.OPERATIONS_PARAMETERS_VECTOR_TEMPLATE_FILENAME),
        )
        self._operation_name_one_hot_encoder = self._load_pickle(
            os.path.join(processed_dir, self.OPERATION_NAME_ONE_HOT_ENCODER_FILENAME),
        )
        self._operations_preprocessors = self._load_pickle(
            os.path.join(processed_dir, self.OPERATIONS_PREPROCESSORS_FILENAME),
        )
        self._metrics_scaler = self._load_pickle(
            os.path.join(processed_dir, self.METRICS_SCALER_FILENAME),
        )

    def _fit_operation_name_one_hot_encoder(self, names: List[str]) -> None:
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoder.fit(np.array(names).reshape(-1, 1))

        path = os.path.join(self.processed_dir, self.OPERATION_NAME_ONE_HOT_ENCODER_FILENAME)
        with open(path, "wb") as f:
            pickle.dump(one_hot_encoder, f)

    def _list_dicts2dict_lists(self, list_dicts: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        keys = set(chain.from_iterable([list(dict.keys()) for dict in list_dicts]))
        dict_lists = {key: [dict[key] for dict in list_dicts if key in dict] for key in keys}
        return dict_lists

    def _fit_operations_preprocessors(self, operations_names: List[str], nodes: List[Dict[str, Any]]) -> None:
        operations_preprocessors = defaultdict(dict)
        operations_parameters_vector_indexes = defaultdict(dict)
        start_index = 0
        for operation_name in operations_names:
            operation_nodes = list(filter(lambda x: x["operation_name"] == operation_name, nodes))
            nodes_parameters = self._list_dicts2dict_lists(self._get_operations_parameters(operation_nodes))
            for parameter_name, parameter_values in nodes_parameters.items():
                parameter_values = np.array(parameter_values)
                if parameter_values.dtype == float or parameter_values.dtype == int:
                    preprocessor = MinMaxScaler()
                else:
                    preprocessor = OneHotEncoder(sparse_output=False)
                preprocessor.fit(parameter_values.reshape(-1, 1))
                operations_preprocessors[operation_name][parameter_name] = preprocessor
                if isinstance(preprocessor, OneHotEncoder):
                    stop_index = start_index + len(preprocessor.categories_[0])
                else:
                    stop_index = start_index + 1
                operations_parameters_vector_indexes[operation_name][parameter_name] = (start_index, stop_index)
                start_index = stop_index

        operations_vector_length = start_index
        operations_parameters_vector_template = np.full(operations_vector_length, fill_value=-1.)
        with open(os.path.join(self.processed_dir, self.OPERATIONS_PREPROCESSORS_FILENAME), "wb") as f:
            pickle.dump(operations_preprocessors, f)
        with open(os.path.join(self.processed_dir, self.OPERATIONS_PARAMETERS_VECTOR_INDEXES_FILENAME), "wb") as f:
            pickle.dump(operations_parameters_vector_indexes, f)
        with open(os.path.join(self.processed_dir, self.OPERATIONS_PARAMETERS_VECTOR_TEMPLATE_FILENAME), "wb") as f:
            pickle.dump(operations_parameters_vector_template, f)

    def _fit_metrics_scaler(self, targets: List[Dict[str, float]]):
        targets = self._list_dicts2dict_lists(targets)
        targets = np.concatenate(list(targets.values()), axis=-1)
        metrics_scaler = MinMaxScaler()
        metrics_scaler.fit(targets)
        with open(os.path.join(self.processed_dir, self.METRICS_SCALER_FILENAME), "wb") as f:
            pickle.dump(metrics_scaler, f)

    def process(self) -> None:
        """Implement method required by `torch_geometric.data.Dataset`."""
        pipelines_nodes = [self._get_nodes_from_json(json) for json in self.json_pipelines]
        pipelines_operations_names = list(
            chain.from_iterable(self._get_operations_names(pipeline_nodes) for pipeline_nodes in pipelines_nodes),
        )
        operations_names = list(set(pipelines_operations_names))
        self._fit_operation_name_one_hot_encoder(operations_names)
        nodes = list(chain.from_iterable(pipelines_nodes))
        self._fit_operations_preprocessors(operations_names, nodes)
        targets = [{k: [v, ] for k, v in self._load_pickle(pickle).items()} for pickle in self.pickle_metrics]
        self._fit_metrics_scaler(targets)

    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """Implement method required by `torch_geometric.data.Dataset`."""
        return (
            self.OPERATION_NAME_ONE_HOT_ENCODER_FILENAME,
            self.OPERATIONS_PREPROCESSORS_FILENAME,
            self.OPERATIONS_PARAMETERS_VECTOR_INDEXES_FILENAME,
            self.OPERATIONS_PARAMETERS_VECTOR_TEMPLATE_FILENAME,
            self.METRICS_SCALER_FILENAME,
        )

    def len(self) -> int:
        """Implement method required by `torch_geometric.data.Dataset`."""
        return len(self.json_pipelines)

    def _load_pickle(self, path: Union[str, Path]) -> Any:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with open(path) as f:
            data = json.load(f)
        return data

    def _get_operations_ids(self, nodes: List[Dict[str, Any]]) -> List[int]:
        return [node["operation_id"] for node in nodes]

    def _get_operations_names(self, nodes: List[Dict[str, Any]], order: List[int] = None) -> List[str]:
        operations_names = []
        if order is None:
            for node in nodes:
                operation_name = node["operation_name"]
                if operation_name is None:  # TODO: is it fix or ok?
                    operation_type = node["operation_type"]
                    operation_name = self.type_name_mapping[operation_type]
                operations_names.append(operation_name)
        else:
            for index in order:
                operation_name = nodes[index]["operation_name"]
                if operation_name is None:
                    operation_type = nodes[index]["operation_type"]
                    operation_name = self.type_name_mapping[operation_type]
                operations_names.append(operation_name)
        return operations_names

    def _get_operations_parameters(self, nodes: List[Dict[str, Any]], order: List[int] = None) -> List[Dict[str, Any]]:
        if order is None:
            return [node["params"] for node in nodes]
        else:
            return [nodes[index]["params"] for index in order]

    def _operation_name2vec(self, operation_name: str) -> np.ndarray:
        return self._operation_name_one_hot_encoder.transform(np.array([[operation_name, ]])).reshape(-1)

    def _metrics_preprocessing(self, metrics: Dict[str, float]) -> torch.Tensor:
        processed = self._metrics_scaler.transform(np.array(list(metrics.values())).reshape(1, -1)).reshape(-1)
        return torch.Tensor(processed)

    def _operation_preprocessing(self, operation_name: str, operation_parameters: Dict[str, Any]) -> Dict[
        str, np.ndarray]:
        processed = {}
        for parameter_name, parameter_value in operation_parameters.items():
            preprocessor = self._operations_preprocessors[operation_name][parameter_name]
            processed[parameter_name] = preprocessor.transform(np.array([[parameter_value, ]])).reshape(-1)
        return processed

    def _parameters2vec(self, operation_name: str, operation_parameters: Dict[str, Any]) -> np.ndarray:
        processed_operation_parameters = self._operation_preprocessing(operation_name, operation_parameters)
        operations_parameters_vector = self._operations_parameters_vector_template.copy()
        for parameter_name, parameter_value in processed_operation_parameters.items():
            start_index, stop_index = self._operations_parameters_vector_indexes[operation_name][parameter_name]
            operations_parameters_vector[start_index: stop_index] = parameter_value
        return operations_parameters_vector

    def _operation2vec(self, operation_name: str, operation_parameters: Dict[str, Any]) -> np.ndarray:
        name_vec = self._operation_name2vec(operation_name)
        if self.use_operations_hyperparameters:
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

    def _get_edge_index_tensor(self, nodes: List[Dict[str, Any]]) -> torch.LongTensor:
        edges = []
        for node in nodes:
            nodes_from = node["nodes_from"]
            if len(nodes_from) > 0:
                target = node["operation_id"]
                for source in nodes_from:
                    if self.direction == "undirected":
                        edges.append([source, target])
                        edges.append([target, source])
                    elif self.direction == "directed":
                        edges.append([source, target])
                    elif self.direction == "reversed":
                        edges.append([target, source])
        return torch.LongTensor(edges).T

    def _get_nodes_from_json(self, json: Path) -> List[Dict[str, Any]]:
        data = self._load_json(json)
        nodes = data["nodes"]
        return nodes

    def _get_data(self, idx: int) -> Data:
        json = self.json_pipelines[idx]
        nodes = self._get_nodes_from_json(json)
        operations_ids = self._get_operations_ids(nodes)
        operations_names = self._get_operations_names(nodes, operations_ids)
        operations_parameters = self._get_operations_parameters(nodes, operations_ids)
        operations_tensor = self._operations2tensor(operations_names, operations_parameters)
        edge_index = self._get_edge_index_tensor(nodes)
        data = Data(operations_tensor, edge_index)
        return data

    def _get_metrics(self, idx: int) -> torch.Tensor:
        pickle = self.pickle_metrics[idx]
        metrics = self._load_pickle(pickle)
        return self._metrics_preprocessing(metrics)

    def get(self, idx: int) -> Tuple[Data, torch.Tensor]:
        """Implement method required by `torch_geometric.data.Dataset`."""
        data = self._get_data(idx)
        metric = self._get_metrics(idx)
        return data, metric
