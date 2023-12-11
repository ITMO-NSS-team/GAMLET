import torch.nn as nn
from meta_automl.surrogate.hetero.misc import NODES_DIMENSIONS, OPERATIONS
from torchvision.ops import MLP
from torch import Tensor
from typing import List

DEFAULT_OUT_DIM = max(list(NODES_DIMENSIONS.values()))


def default_hyperparams_encoder_builder(input_dim: int, out_dim: int = DEFAULT_OUT_DIM) -> nn.Module:
    encoder = MLP(
        in_channels=input_dim,
        hidden_channels=[out_dim] * 2,
        dropout=0.1,
        inplace=False,
    )
    encoder.out_dim = out_dim
    return encoder

class Encoder(nn.Module):
    def __init__(self, n_input: int,  out_dim: int, n_hidden: List[int]):
        super().__init__()
        assert len(n_hidden) > 0
        self.model = nn.Sequential()
        self.model.append(nn.Linear(n_input, n_hidden[0]))
        self.model.append(nn.ReLU())
        for i in range(1, len(n_hidden)):
             self.model.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
             self.model.append(nn.ReLU())
        self.model.append(nn.Linear(n_hidden[-1], out_dim))
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
def pretrained_hyperparams_encoder_builder(input_dim: int, out_dim: int = DEFAULT_OUT_DIM) -> nn.Module:
    return Encoder(
        n_input=input_dim, 
        n_hidden=[out_dim] * 2, 
        out_dim=out_dim,
    )
