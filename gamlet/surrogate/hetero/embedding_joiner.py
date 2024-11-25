import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

class CatEmbeddingJoiner(nn.Module):
    def __init__(
        self,
        op_name_embedding_dim: int,
        op_hyperparams_embedding_dim: int,
    ):
        super().__init__()
        self.out_dim = op_name_embedding_dim + op_hyperparams_embedding_dim

    def forward(self, op_name_embedding: Dict[str, Tensor], op_hyperparams_embedding: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {k: torch.cat([op_name_embedding[k], op_hyperparams_embedding[k]], dim=1) for k in op_name_embedding.keys()}
