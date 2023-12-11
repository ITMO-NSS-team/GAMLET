# -*- coding: utf-8 -*-
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import aggr
from torch_geometric.nn.inits import reset


class CustormAggregation(aggr.Aggregation):
    def __init__(
        self,
        local_nn: Optional[torch.nn.Module] = None,
        global_nn: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def reset_parameters(self):
        if self.local_nn is not None:
            reset(self.local_nn)
        if self.global_nn is not None:
            reset(self.global_nn)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        if self.local_nn is not None:
            x = self.local_nn(x)
        x = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        if self.global_nn is not None:
            x = self.global_nn(x)
        return x


class MLPDatasetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout_in: float = 0.4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.inp_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=dropout_in),
            nn.Linear(input_dim, hidden_dim),
        )

        self.block1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.out_dim = output_dim

    def forward(self, data: Data) -> Tensor:
        z = self.inp_layer(data.x)
        z = self.block1(z)
        z = self.block2(z)
        return z


class ColumnDatasetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.inp_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        mlp1 = nn.Sequential(nn.BatchNorm1d(input_dim), nn.Linear(input_dim, hidden_dim), nn.ReLU())
        mlp2 = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        agg_f = CustormAggregation(mlp1, mlp2)
        self.multi_aggr = aggr.MultiAggregation(aggrs=["mean", agg_f], mode="cat")
        self.out_dim = input_dim + hidden_dim

    def forward(self, data: Data) -> Tensor:
        z = data.x
        z = self.multi_aggr(z, ptr=data.ptr, dim=0)
        return z
