# -*- coding: utf-8 -*-
from typing import Union

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor, nn
from torch_geometric.data import Batch

from .gnn_layers import get_simple_gnn_layer


class SimpleGNNEncoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        d_model: int,
        gnn_type: str = "gcn",
        dropout: float = 0.0,
        num_layers: int = 4,
        batch_norm: bool = False,
        in_embed: bool = True,
        max_seq_len: int = None,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = num_layers
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.dropout = dropout
        self.batch_norm = batch_norm

        layers = []
        batch_layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, d_model, **kwargs))
            if self.batch_norm:
                batch_layers.append(nn.BatchNorm1d(d_model))

        self.gnn = nn.ModuleList(layers)
        self.batch_norms = nn.ModuleList(batch_layers)

        self.pooling = gnn.global_mean_pool

        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size, out_features=d_model, bias=False)
        self.max_seq_len = max_seq_len
        self.out_dim = d_model

    def forward(self, data: Batch) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if hasattr(data, "node_depth"):
            node_depth = data.node_depth
            output = self.embedding(x, node_depth.view(-1))
        else:
            output = self.embedding(x)

        for layer in range(self.num_layers):
            output = self.gnn[layer](output, edge_index)
            if self.batch_norm:
                output = self.batch_norms[layer](output)

            if layer == self.num_layers - 1:
                output = F.dropout(output, self.dropout)
            else:
                output = F.dropout(F.relu(output), self.dropout)

        # readout step
        output = self.pooling(output, batch)

        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list

        return output
