from collections import defaultdict
from typing import Dict, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch


class HeterogeneousData:
    def __init__(self, edge_index: Tensor, node_idxes_per_type: Dict[str, Sequence[int]], hparams: Dict[str, Tensor] = {}, encoded_type: Dict[str, Tensor] = {}):
        # kwargs should be like `a = torch.rand(2, 4), b = torch.rand(3, 9), ...`
        self.edge_index = edge_index
        self.node_idxes_per_type = {k: torch.LongTensor(v) for k, v in node_idxes_per_type.items()}
        self.hparams = {}
        self.encoded_type = {}
        self.num_nodes = 0
        keys = set(list(hparams.keys()) + list(encoded_type.keys()))
        for k in keys:
            num_nodes = None
            try:
                self.hparams[k] = hparams[k]
                num_nodes = hparams[k].shape[0]
            except KeyError:
                pass
            try:
                self.encoded_type[k] = encoded_type[k]
                num_nodes = encoded_type[k].shape[0]
            except KeyError:
                pass
            self.num_nodes += num_nodes


class HeterogeneousBatch:
    def __init__(self):
        self.batch: Tensor = None
        self.edge_index: Tensor = None
        self.node_idxes_per_type: Dict[str, Tensor] = None
        self.hparams: Dict[str, Tensor] = None
        self.encoded_type: Dict[str, Tensor] = None
        self.num_nodes: int = None

    @staticmethod
    def from_heterogeneous_data_list(data_list: Sequence[HeterogeneousData]) -> "HeterogeneousBatch":
        total_nodes = 0
        batch = []
        edge_index = []
        hparams = defaultdict(list)
        encoded_type = defaultdict(list)
        node_idxes_per_type = defaultdict(list)

        for i, data in enumerate(data_list):
            batch.extend([i for _ in range(data.num_nodes)])

            for node_type, node_index in data.node_idxes_per_type.items():
                node_idxes_per_type[node_type].extend(node_index + total_nodes)

            data_edge_index = data.edge_index + total_nodes
            edge_index.append(data_edge_index)
            total_nodes += data.num_nodes

            for k, v in data.hparams.items():
                hparams[k].append(v)
            for k, v in data.encoded_type.items():
                encoded_type[k].append(v)

        my_batch = HeterogeneousBatch()
        my_batch.batch = torch.LongTensor(batch)
        my_batch.node_idxes_per_type = {k: torch.LongTensor(v) for k, v in node_idxes_per_type.items()}
        my_batch.edge_index = torch.hstack(edge_index)
        my_batch.hparams = {k: torch.vstack(v) for k, v in hparams.items()}
        my_batch.encoded_type = {k: torch.vstack(v) for k, v in encoded_type.items()}
        my_batch.num_nodes = total_nodes
        return my_batch

    def to_pyg_batch(self, dim: int, transformed: Dict[str, Tensor]) -> Batch:
        x = torch.empty(self.num_nodes, dim).to(self.batch.device)
        for node_type, node_data in transformed.items():
            idxes = self.node_idxes_per_type[node_type]
            x[idxes] = node_data
        pyg_batch = Batch(x=x, batch=self.batch, edge_index=self.edge_index)
        return pyg_batch

    def __getitem__(self, key: str) -> Dict[str, Tensor]:
        res = {}
        try:
            res["encoded_type"] = self.encoded_type[key]
        except KeyError:
            pass
        try:
            res["hparams"] = self.hparams[key]
        except KeyError:
            pass
        return res
