from collections import OrderedDict
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from automl_surrogate.models.hyperparams_encoder import (
    DEFAULT_OUT_DIM,
    default_hyperparams_encoder_builder,
    pretrained_hyperparams_encoder_builder,
)
from automl_surrogate.models.misc import NODES_DIMENSIONS, OPERATIONS


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
        embeddings = {}
        for op_name, op_params in x.items():
            if op_name in self.encoders:
                embedding = self.encoders[op_name](op_params)
                embeddings[op_name] = embedding
            elif op_name in self.learnable:
                embeddings[op_name] = self.learnable[op_name].repeat(op_params.shape[0], 1)
            else:
                raise ValueError(f"Not specified operation type: {op_name}.")
        return embeddings


class PretrainedHyperparametersEmbedder(HyperparametersEmbedder):
    def __init__(
        self,
        autoencoder_ckpt_path: str,
        out_dim: Optional[int] = DEFAULT_OUT_DIM,
        encoder_builder: Optional[Callable] = pretrained_hyperparams_encoder_builder,
        operations: Optional[Sequence[str]] = OPERATIONS,
        trainable: bool = False,
    ):
        super().__init__()
        self.trainable = trainable
        self.out_dim = out_dim

        state_dict = torch.load(autoencoder_ckpt_path, map_location="cpu")["state_dict"]

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
        self.load_pretarained_weights(state_dict)

    def load_pretarained_weights(self, state_dict: OrderedDict) -> None:
        updated_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("learnable"):
                if len(v.shape) == 1:
                    v = v.reshape(1, -1)
                updated_state_dict[k] = v
            elif k.startswith("autoencoders"):
                if ".encoder" in k:
                    k = k.replace(".encoder", "")
                    k = k.replace("autoencoders.", "encoders.")
                    updated_state_dict[k] = v
            else:
                raise ValueError(f"Unrecognized key: {k}")
        self.load_state_dict(updated_state_dict, strict=False)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.trainable:
            return super().forward(x)
        else:
            with torch.no_grad():
                return super().forward(x)


class TransitHyperparametersEmbedder:
    def __init__(self, out_dim: int = DEFAULT_OUT_DIM):
        self.out_dim = out_dim

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return x

    def __call__(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.forward(x)
