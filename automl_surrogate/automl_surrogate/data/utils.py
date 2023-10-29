from typing import Sequence

import torch
from torch import Tensor

from .data_types import HeterogeneousBatch


def transfer_batch_to_device(batch: Sequence, device: torch.device, dataloader_idx: int) -> Sequence:
    res = []
    for e in batch:
        if isinstance(e, Tensor):
            res.append(e.to(device))
        elif isinstance(e, HeterogeneousBatch):
            e.batch = e.batch.to(device)
            e.ptr = e.ptr.to(device)
            e.edge_index = e.edge_index.to(device)
            e.node_idxes_per_type = {k: v.to(device) for k, v in e.node_idxes_per_type.items()}
            e.hparams = {k: v.to(device) for k, v in e.hparams.items()}
            e.encoded_type = {k: v.to(device) for k, v in e.encoded_type.items()}
            res.append(e)
        else:
            raise TypeError(f"Uknown type f{type(e)}")
    return res
