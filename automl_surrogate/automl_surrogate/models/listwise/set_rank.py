import torch.nn as nn
from torch import Tensor


class SetRank(nn.Module):
    def __init__(self, in_dim: int, nhead: int, dim_feedforward: int, dropout: int, num_layers: int):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.mhsa_block = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(in_dim, 1)
        self.out_dim = 1

    def forward(self, candidates: Tensor) -> Tensor:
        # Expected input shape is [BATCH, N, HIDDEN]
        mhsa_output = self.mhsa_block(candidates)
        scores = self.linear(mhsa_output).squeeze(2)
        # Output shape is [BATCH, N]
        return scores
