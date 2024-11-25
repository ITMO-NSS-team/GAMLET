import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DirectRanker(nn.Module):
    def __init__(
        self,
        in_dim: int,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=False)
        self.out_dim = 1

    def forward(self, candidate_1: Tensor, candidate_2: Tensor) -> Tensor:
        difference = candidate_1 - candidate_2
        score = F.tanh(self.linear(difference) / 2)
        return score


class FusionDirectRanker(nn.Module):
    def __init__(
        self,
        in_dim: int,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=False)
        self.out_dim = 1

    def forward(self, candidate_1: Tensor, candidate_2: Tensor, dataset: Tensor) -> Tensor:
        difference = candidate_1 - candidate_2
        fused = torch.hstack([difference, dataset])
        score = F.tanh(self.linear(fused) / 2)
        return score
