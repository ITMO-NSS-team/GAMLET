# from automl_surrogate.models.heterogeneous.misc import OPERATIONS
from typing import Optional  # , List

import torch.nn as nn
from torch import Tensor


class NameEmbedder(nn.Module):
    def __init__(self, out_dim: Optional[int] = 2, num_embeddings: Optional[int] = 100):
        super().__init__()
        self.out_dim = out_dim
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x).squeeze(1)
