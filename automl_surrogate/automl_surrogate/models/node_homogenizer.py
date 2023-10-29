from typing import Any, Dict, Optional

import torch.nn as nn
from torch import Tensor

from automl_surrogate.data import HeterogeneousBatch
from automl_surrogate.models import embedding_joiner, hyperparams_embedder, name_embedder


class NodeHomogenizer(nn.Module):
    def __init__(
        self,
        op_hyperparams_embedder: Optional[nn.Module] = None,
        op_name_embedder: Optional[nn.Module] = None,
        emebedding_joiner: Optional[nn.Module] = None,
    ) -> None:
        if op_hyperparams_embedder is None and op_name_embedder is None:
            err_msg = "At least one of `op_hyperparams_embedder` and `op_name_embedder` must be provided."
            raise ValueError(err_msg)
        if (op_hyperparams_embedder is None or op_name_embedder is None) and emebedding_joiner is not None:
            err_msg = "`emebedding_joiner` can only be used when both `op_hyperparams_embedder` and `op_name_embedder` are provided."
            raise ValueError(err_msg)
        if (op_hyperparams_embedder is not None and op_name_embedder is not None) and emebedding_joiner is None:
            err_msg = "`emebedding_joiner` must be provided when both `op_hyperparams_embedder` and `op_name_embedder` are provided."
            raise ValueError(err_msg)
        super().__init__()
        self.op_hyperparams_embedder = op_hyperparams_embedder
        self.op_name_embedder = op_name_embedder
        self.emebedding_joiner = emebedding_joiner

        if self.emebedding_joiner is not None:
            self.out_dim = self.emebedding_joiner.out_dim
        elif self.op_name_embedder is not None:
            self.out_dim = self.op_name_embedder.out_dim
        else:
            self.out_dim = self.op_hyperparams_embedder.out_dim

    def forward(
        self,
        x: HeterogeneousBatch,
    ) -> Dict[str, Tensor]:
        """A single operation data is expected. Expected batch size is equal to 1."""
        hparams_embed: Dict[str, Tensor] = self.op_hyperparams_embedder(x.hparams)
        type_embed: Dict[str, Tensor] = {k: self.op_name_embedder(v) for k, v in x.encoded_type.items()}
        embeddings: Dict[str, Tensor] = self.emebedding_joiner(hparams_embed, type_embed)
        return embeddings
        # if self.emebedding_joiner is not None:
        #     op_name_embedding: Tensor = self.op_name_embedder(op_name_vec)
        #     op_hyperparams_embedding: Tensor = self.op_hyperparams_embedder({op_name: op_hyperparams_vec})[op_name]
        #     embeddings: Tensor = self.emebedding_joiner(op_name_embedding, op_hyperparams_embedding)
        # elif self.op_name_embedder is not None:
        #     embeddings: Tensor = self.op_name_embedder(op_name_vec)
        # else:
        #     embeddings: Tensor = self.op_hyperparams_embedder(x.hparams)
        # return embeddings


def build_node_homogenizer(model_parameters: Dict[str, Any]) -> NodeHomogenizer:
    op_hyperparams_embedder = None
    if "op_hyperparams_embedder" in model_parameters:
        config = model_parameters["op_hyperparams_embedder"]
        class_ = getattr(hyperparams_embedder, config["class"])
        op_hyperparams_embedder = class_(**{k: v for k, v in config.items() if k != "class"})
    op_name_embedder = None
    if "op_name_embedder" in model_parameters:
        config = model_parameters["op_name_embedder"]
        class_ = getattr(name_embedder, config["class"])
        op_name_embedder = class_(**{k: v for k, v in config.items() if k != "class"})
    emebedding_joiner = None
    if "embedding_joiner" in model_parameters:
        config = model_parameters["embedding_joiner"]
        class_ = getattr(embedding_joiner, config["class"])
        emebedding_joiner = class_(
            op_name_embedding_dim=op_name_embedder.out_dim,
            op_hyperparams_embedding_dim=op_hyperparams_embedder.out_dim,
            **{k: v for k, v in config.items() if k != "class"},
        )
    return NodeHomogenizer(op_hyperparams_embedder, op_name_embedder, emebedding_joiner)
