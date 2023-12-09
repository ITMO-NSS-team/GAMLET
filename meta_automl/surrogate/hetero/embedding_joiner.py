import torch
import torch.nn as nn
from torch import Tensor


class CatEmbeddingJoiner(nn.Module):
    def __init__(
        self,
        op_name_embedding_dim: int,
        op_hyperparams_embedding_dim: int,
    ):
        super().__init__()
        self.out_dim = op_name_embedding_dim + op_hyperparams_embedding_dim

    def forward(self, op_name_embedding: Tensor, op_hyperparams_embedding: Tensor) -> Tensor:
        return torch.cat([op_name_embedding, op_hyperparams_embedding], dim=1)
