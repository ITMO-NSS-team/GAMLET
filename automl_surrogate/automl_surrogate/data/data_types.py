import json
from collections import defaultdict
from typing import Dict, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch


class HeterogeneousData:
    def __init__(
        self,
        edge_index: Tensor = None,
        node_idxes_per_type: Dict[str, Sequence[int]] = {},
        hparams: Dict[str, Tensor] = {},
        encoded_type: Dict[str, Tensor] = {},
        skip_init_: bool = False,
    ):
        if skip_init_:
            self.edge_index = None
            self.node_idxes_per_type = None
            self.hparams = None
            self.encoded_type = None
            self.num_nodes = None

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

    @staticmethod
    def from_json(path: str) -> "HeterogeneousData":
        with open(path, "r") as f:
            data = json.load(f)
        instance = HeterogeneousData(skip_init_=True)
        new = dict()
        for key, value in data.items():
            if key == "edge_index":
                new[key] = torch.LongTensor(value)
            elif key == "node_idxes_per_type" or key == "encoded_type":
                temp = {}
                for key_1, value_1 in value.items():
                    temp[key_1] = torch.LongTensor(value_1)
                new[key] = temp
            elif key == "hparams":
                temp = {}
                for key_1, value_1 in value.items():
                    temp[key_1] = torch.FloatTensor(value_1)
                new[key] = temp
            elif key == "num_nodes":
                new[key] = value
            else:
                raise KeyError(f"Unknown key {key}")
        instance.__dict__.update(new)
        return instance

    def to_json(self, path: str):
        new = dict()
        for key, value in self.__dict__.items():
            if key == "edge_index":
                new[key] = value.numpy().tolist()
            elif key == "node_idxes_per_type" or key == "encoded_type" or key == "hparams":
                temp = {}
                for key_1, value_1 in value.items():
                    temp[key_1] = value_1.numpy().tolist()
                new[key] = temp
            elif key == "num_nodes":
                new[key] = value
            else:
                raise KeyError(f"Unknown key {key}")

        with open(path, "w") as f:
            json.dump(new, f)


class HeterogeneousBatch:
    def __init__(self):
        self.batch: Tensor = None
        self.ptr: Tensor = None
        self.edge_index: Tensor = None
        self.node_idxes_per_type: Dict[str, Tensor] = None
        self.hparams: Dict[str, Tensor] = None
        self.encoded_type: Dict[str, Tensor] = None
        self.num_nodes: int = None

    @staticmethod
    def from_heterogeneous_data_list(data_list: Sequence[HeterogeneousData]) -> "HeterogeneousBatch":
        total_nodes = 0
        batch = []
        ptr = [0]
        edge_index = []
        hparams = defaultdict(list)
        encoded_type = defaultdict(list)
        node_idxes_per_type = defaultdict(list)

        for i, data in enumerate(data_list):
            batch.extend([i for _ in range(data.num_nodes)])
            ptr.append(ptr[-1] + data.num_nodes)
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
        my_batch.ptr = torch.LongTensor(ptr)
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
        pyg_batch = Batch(x=x, batch=self.batch, edge_index=self.edge_index, ptr=self.ptr)
        # pyg_batch_ = Batch(x=x, batch=self.batch, edge_index=self.edge_index)
        # data_list = [self._get_sample(pyg_batch_, i) for i in self.batch.unique()]
        # pyg_batch = Batch.from_data_list(data_list)
        return pyg_batch

    # @staticmethod
    # def _get_sample(batch: Batch, index: int) -> Data:
    #     with torch.no_grad():
    #         idxes = torch.where(batch.batch == index)
    #         p_min = idxes[0].min()
    #         p_max = idxes[0].max()
    #         s1 = batch.edge_index >= p_min
    #         s2 = batch.edge_index <= p_max
    #         s = s1 * s2
    #         edge_index = batch.edge_index[s].reshape(2, -1)
    #         edge_index -= p_min
    #     x = batch.x[idxes]
    #     data = Data(x=x, edge_index=edge_index)
    #     return data

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


# # For GNN pretraining
# from torch.nested import nested_tensor
# from collections import defaultdict
# from typing import Dict, Sequence

# import torch
# from torch import Tensor
# from torch_geometric.data import Batch
# import json


# class HeterogeneousData:
#     def __init__(
#         self,
#         edge_index: "LongTensor",
#         node_idxes_per_type: "Dict[str, Sequence[int]]",
#         hparams: "nested_tensor" = None,
#         encoded_type: "LongTensor" = None,
#         num_nodes: int = None
#     ):
#         if hparams is None and encoded_type is None:
#             raise ValueError(f"No hparams or encoded_type provided")

#         if hparams is not None and encoded_type is not None:
#             if hparams.size(0) != encoded_type.size(0):
#                 raise ValueError(f"{hparams.size(0)=} != {encoded_type.size(0)=}")

#         if num_nodes is not None:
#             self.num_nodes = num_nodes
#         elif hparams is not None:
#             self.num_nodes = hparams.size(0)
#         elif encoded_type is not None:
#             self.num_nodes = encoded_type.size(0)

#         self.edge_index = edge_index
#         self.node_idxes_per_type = {k: torch.LongTensor(v) for k, v in node_idxes_per_type.items()}
#         self.hparams = hparams
#         self.encoded_type = encoded_type

#     def get_hparams(self, name: str):
#         idxes = self.node_idxes_per_type[name]
#         unbind = self.hparams.unbind()
#         return torch.vstack([unbind[i] for i in idxes])

#     def get_encoded_type(self, name: str):
#         idxes = self.node_idxes_per_type[name]
#         return self.encoded_type[idxes]

# class HeterogeneousBatch:
#     def __init__(self):
#         self.batch: "LongTensor" = None
#         self.ptr: "LongTensor" = None
#         self.edge_index: "LongTensor" = None
#         self.node_idxes_per_type: "Dict[str, Tensor]" = None
#         self.hparams:"nested_tensor" = None
#         self.encoded_type: "LongTensor" = None
#         self.num_nodes: int = None
#         self.edge_split: "List[int]" = None
#         self.nodes_before: "List[int]" = None

#     @staticmethod
#     def from_heterogeneous_data_list(data_list: "Sequence[HeterogeneousData]") -> "HeterogeneousBatch":
#         total_nodes = 0
#         batch = []
#         ptr = [0]
#         edge_index = []
#         edge_split = [0]
#         nodes_before = []
#         hparams = []
#         encoded_type = []
#         node_idxes_per_type = defaultdict(list)

#         for i, data in enumerate(data_list):
#             batch.extend([i for _ in range(data.num_nodes)])
#             ptr.append(ptr[-1] + data.num_nodes)
#             for node_type, node_index in data.node_idxes_per_type.items():
#                 node_idxes_per_type[node_type].extend(node_index + total_nodes)

#             edge_split.append(edge_split[-1] + data.edge_index.shape[1])
#             nodes_before.append(total_nodes)

#             data_edge_index = data.edge_index + total_nodes
#             edge_index.append(data_edge_index)
#             total_nodes += data.num_nodes

#             hparams.extend(data.hparams.unbind())
#             encoded_type.append(data.encoded_type)

#         my_batch = HeterogeneousBatch()
#         my_batch.batch = torch.LongTensor(batch)
#         my_batch.ptr = torch.LongTensor(ptr)
#         my_batch.node_idxes_per_type = {k: torch.LongTensor(v) for k, v in node_idxes_per_type.items()}
#         my_batch.edge_index = torch.hstack(edge_index)
#         my_batch.hparams = nested_tensor(hparams)
#         my_batch.encoded_type = torch.hstack(encoded_type)  # vstack if encoded_type.shape is (N, 1)
#         my_batch.num_nodes = total_nodes
#         my_batch.edge_split = edge_split
#         my_batch.nodes_before = nodes_before
#         return my_batch

#     def to_pyg_batch(self, dim: int, transformed: "Dict[str, Tensor]") -> "Batch":
#         x = torch.empty(self.num_nodes, dim).to(self.batch.device)
#         for node_type, node_data in transformed.items():
#             idxes = self.node_idxes_per_type[node_type]
#             x[idxes] = node_data
#         pyg_batch = Batch(x=x, batch=self.batch, edge_index=self.edge_index, ptr=self.ptr)
#         # pyg_batch_ = Batch(x=x, batch=self.batch, edge_index=self.edge_index)
#         # data_list = [self._get_sample(pyg_batch_, i) for i in self.batch.unique()]
#         # pyg_batch = Batch.from_data_list(data_list)
#         return pyg_batch

#     # @staticmethod
#     # def _get_sample(batch: Batch, index: int) -> Data:
#     #     with torch.no_grad():
#     #         idxes = torch.where(batch.batch == index)
#     #         p_min = idxes[0].min()
#     #         p_max = idxes[0].max()
#     #         s1 = batch.edge_index >= p_min
#     #         s2 = batch.edge_index <= p_max
#     #         s = s1 * s2
#     #         edge_index = batch.edge_index[s].reshape(2, -1)
#     #         edge_index -= p_min
#     #     x = batch.x[idxes]
#     #     data = Data(x=x, edge_index=edge_index)
#     #     return data

#     def get_hparams(self, name: str):
#         idxes = self.node_idxes_per_type[name]
#         unbind = self.hparams.unbind()
#         return torch.vstack([unbind[i] for i in idxes])

#     def get_encoded_type(self, name: str):
#         idxes = self.node_idxes_per_type[name]
#         return self.encoded_type[idxes]

#     def __getitem__(self, key: str) ->" Dict[str, Tensor]":
#         res = {}
#         idxes = self.node_idxes_per_type[name]
#         try:
#             res["encoded_type"] = self.encoded_type[idxes]
#         except KeyError:
#             pass
#         try:
#             unbind = self.hparams.unbind()
#             res["hparams"] = torch.vstack([unbind[i] for i in idxes])
#         except KeyError:
#             pass
#         return res
