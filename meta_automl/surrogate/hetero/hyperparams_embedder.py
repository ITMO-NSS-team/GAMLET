from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from meta_automl.surrogate.hetero.hyperparams_encoder import default_hyperparams_encoder_builder, DEFAULT_OUT_DIM
from meta_automl.surrogate.hetero.misc import NODES_DIMENSIONS, OPERATIONS


class HyperparametersEmbedder(nn.Module):
    def __init__(
        self,
        out_dim: Optional[int] = DEFAULT_OUT_DIM,
        encoder_builder: Optional[Callable] = default_hyperparams_encoder_builder,
        operations: Optional[Sequence[str]] = OPERATIONS,
    ):
        super().__init__()
        self.out_dim = out_dim

        encoders = {}
        learnables = {}

        for op_name in operations:
            try:
                input_dim = NODES_DIMENSIONS[op_name]
            except KeyError:
                error_msg = (
                    f"Unsupported operation type: {op_name}. "
                    f"See list of supported operations in `meta_automl.surrogate.hetero_surrogate.misc`"
                )
                print(error_msg)
            if input_dim > 0:
                encoders[op_name] = encoder_builder(input_dim, self.out_dim)
            else:
                learnables[op_name] = nn.Parameter(data=torch.rand(1, self.out_dim), requires_grad=True)

        self.encoders = nn.ModuleDict(encoders)
        self.learnable = nn.ParameterDict(learnables)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Expected batch size is equal to 1."""
        embeddings = {}
        for op_name in x:
            if op_name in self.encoders:
                embedding = self.encoders[op_name](x[op_name])
                embeddings[op_name] = embedding
            elif op_name in self.learnable:
                embeddings[op_name] = self.learnable[op_name]
            else:
                raise ValueError(f"Not specified operation type: {op_name}.")
        return embeddings
