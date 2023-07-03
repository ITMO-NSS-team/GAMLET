import os
import pickle
import random
from pathlib import Path
from typing import Union, List, Tuple, Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.data.data import Data
from tqdm import tqdm


class SyntheticPipelineDataset(Dataset):
    """
    Generally follows HomogeneousPipelineDataset.
    Generates synthetic pipelines with metric being a deternized function.
    If it is useful, docs will be added. Otherwise, the class will be deleted.
    """
    DEFAULT_NUM_PIPELINE_NODE_TYPES = 15
    DEFAULT_PIPELINE_LENGTH = 9
    DEFAULT_DATASET_LENGTH = 10000
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15

    # undirected, directed, reversed
    def __init__(
            self,
            root: str,
            split: str = None,  # train, val, test
            log: bool = True,
            direction: str = "undirected",  # undirected, directed, reversed
            dataset_len: int = None,
            num_pipeline_node_types: int = None,
            max_pipeline_length: int = None,
            mode: str = "offline",  # online
    ):
        self.mode = mode
        self.direction = direction
        self.sample_idxes = np.arange(dataset_len or self.DEFAULT_DATASET_LENGTH)
        self.num_pipeline_node_types = num_pipeline_node_types or self.DEFAULT_NUM_PIPELINE_NODE_TYPES
        self.max_pipeline_length = max_pipeline_length or self.DEFAULT_PIPELINE_LENGTH
        self.data_dir = f"{self.num_pipeline_node_types}_{self.max_pipeline_length}_data"
        self.metrics_dir = f"{self.num_pipeline_node_types}_{self.max_pipeline_length}_metrics"
        self.edge_weights_fname = f"{self.num_pipeline_node_types}_{self.max_pipeline_length}_edge_weights.pickle"
        super().__init__(root, log=log)
        if split is not None:
            if split == "train":
                start = 0
                stop = int(self.TRAIN_RATIO * len(self.sample_idxes))
            elif split == "val":
                start = int(self.TRAIN_RATIO * len(self.sample_idxes))
                stop = int((self.TRAIN_RATIO + self.VAL_RATIO) * len(self.sample_idxes))
            elif split == "test":
                start = int((self.TRAIN_RATIO + self.VAL_RATIO) * len(self.sample_idxes))
                stop = len(self.sample_idxes)
            else:
                raise ValueError(f"Unknown split: {split}")
            self.sample_idxes = self.sample_idxes[start: stop]

        self.edge_weights = self._load_pickle(os.path.join(self.raw_dir, self.edge_weights_fname))
        # TODO: add path from node as extra parameter
        self.norm = sum([self.edge_weights.reshape(-1).max()] * (self.max_pipeline_length - 1))  # Max metric value

    def _load_pickle(self, path: Union[str, Path]) -> Any:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def generate_edge_weights(self) -> np.ndarray:
        return np.random.rand(self.num_pipeline_node_types, self.num_pipeline_node_types)

    def generate_directed_random_tree(self, length: int) -> nx.DiGraph:
        """Generate a random directed tree with n nodes and reverse it"""
        tree = nx.DiGraph()
        tree.add_node(0)
        if length == 1:
            return tree
        # Generate a random directed tree by adding edges to an empty graph
        for i in range(1, length):
            # Randomly select a parent node from nodes already in the tree
            parent = random.choice(list(tree.nodes()))
            # Add the new node and edge to the tree
            tree.add_node(i)
            tree.add_edge(parent, i)
        return tree.reverse()

    def get_metric(self, pipeline_graph: nx.DiGraph, node_types: np.ndarray) -> torch.Tensor:
        metric = 0
        for edge in pipeline_graph.edges():
            src, tgt = edge
            src_type = node_types[src]
            tgt_type = node_types[tgt]
            weight = self.edge_weights[src_type, tgt_type]
            metric += weight
        return torch.tensor([metric / self.norm], dtype=torch.float32)

    def generate_sample(self) -> Tuple[Data, torch.Tensor]:
        pipeline_length = np.random.randint(2, self.max_pipeline_length)
        one_hot_nodes_types = []
        node_types = np.random.choice(np.arange(self.num_pipeline_node_types), pipeline_length, replace=True)
        for node_type in node_types:
            one_hot_vector = torch.zeros(self.num_pipeline_node_types)
            one_hot_vector[node_type] = 1
            one_hot_nodes_types.append(one_hot_vector)
        one_hot_nodes_types = torch.vstack(one_hot_nodes_types)
        pipeline_graph = self.generate_directed_random_tree(pipeline_length)
        edge_index = torch.LongTensor(list(pipeline_graph.edges())).T
        metric = self.get_metric(pipeline_graph, node_types)
        return Data(one_hot_nodes_types, edge_index), metric

    def load_sample(self, idx: int) -> Tuple[Data, torch.Tensor]:
        data = self._load_pickle(os.path.join(self.raw_dir, self.data_dir, f"{idx}.pickle"))
        metric = self._load_pickle(os.path.join(self.raw_dir, self.metrics_dir, f"{idx}.pickle"))
        return data, metric

    def get(self, idx: int) -> Tuple[Data, torch.Tensor]:
        if self.mode == "online":
            data, metric = self.generate_sample()
        elif self.mode == "offline":
            data, metric = self.load_sample(idx)

        if self.direction == "reversed":
            data.edge_index = torch.flip(data.edge_index, dims=[0, ])
        elif self.direction == "undirected":
            data.edge_index = torch.hstack([data.edge_index, torch.flip(data.edge_index, dims=[0, ])])

        return data, metric

    def len(self) -> int:
        return len(self.sample_idxes)

    def download(self) -> None:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        self.edge_weights = self.generate_edge_weights()
        self.norm = sum([self.edge_weights.reshape(-1).max()] * (self.max_pipeline_length - 1))  # Max metric value

        with open(os.path.join(self.raw_dir, self.edge_weights_fname), "wb") as f:
            pickle.dump(self.edge_weights, f)

        if not os.path.exists(os.path.join(self.raw_dir, self.data_dir)):
            os.mkdir(os.path.join(self.raw_dir, self.data_dir))
        if not os.path.exists(os.path.join(self.raw_dir, self.metrics_dir)):
            os.mkdir(os.path.join(self.raw_dir, self.metrics_dir))

        for idx in tqdm(self.sample_idxes):
            data, metric = self.generate_sample()
            with open(os.path.join(self.raw_dir, self.data_dir, f"{idx}.pickle"), "wb") as f:
                pickle.dump(data, f)
            with open(os.path.join(self.raw_dir, self.metrics_dir, f"{idx}.pickle"), "wb") as f:
                pickle.dump(metric, f)

    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return (
            self.data_dir,
            self.metrics_dir,
            self.edge_weights_fname,
        )


if __name__ == "__main__":
    dataset = SyntheticPipelineDataset(r"C:\Users\Konstantin\PycharmProjects\NIR\dataset\synthetic_pipeline_dataset")
