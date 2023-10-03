from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.models import MLP, GCN
from torch_geometric.utils import scatter


# TODO: this class is subject of discussion. This model do not work well.
class HomogeneousGCN(nn.Module):
    # `aggregation` = `last` means that a pipeline last node embedding is taken as the pipeline embedding.
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            gnn_hidden_channels: int = 32,
            gnn_num_layers: int = 3,
            mlp_hidden_channels: int = 32,
            mlp_num_layers: int = 3,
            aggregation: str = "sum",  # `sum`, `max`, `mean`, `last` is supported.
            clip_output: Tuple[float, float] = None,
    ):
        super().__init__()
        self.encoder = GCN(
            in_channels=in_channels,
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            norm="batch_norm"
        )
        self.mlp = MLP(
            in_channels=self.encoder.out_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_num_layers,
            act="relu",
            norm="batch_norm",
            plain_last=True,
        )
        self.aggregation = aggregation
        self.clip_output = clip_output

    def forward(self, data: Data) -> torch.Tensor:
        z = self.encoder(data.x, data.edge_index)
        if self.aggregation == "last":
            df = pd.DataFrame(data=data.batch.cpu(), columns=["batch", ])
            idxes = df.groupby("batch").apply(lambda x: x.index[-1]).to_numpy()
            z = z[idxes]  # Aggregate last node
        else:
            z = scatter(z, data.batch, dim=0, reduce=self.aggregation)
        if self.clip_output is None:
            return self.mlp(z)
        else:
            return torch.clip_(self.mlp(z), *self.clip_output)
