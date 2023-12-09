import torch.nn as nn
from meta_automl.surrogate.hetero.misc import NODES_DIMENSIONS, OPERATIONS
from torchvision.ops import MLP

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
