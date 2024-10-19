import torch
import torch.nn as nn
from torch import Tensor


def bubble_argsort(
    comparator: nn.Module, pipe_embs: Tensor, device: torch.device, dataset_embeddings: Tensor = None
) -> Tensor:
    # Batch is 1. pipe_embs shape is [BATCH, N, HIDDEN]
    batch_size = pipe_embs.shape[0]
    n = pipe_embs.shape[1]
    indices = torch.LongTensor([list(range(n))] * batch_size).to(device)
    for i in range(n - 1):
        swapped = False
        for j in range(n - i - 1):
            with torch.no_grad():
                # whether j-th element is better than (j+1)-th element
                if dataset_embeddings is None:
                    to_swap = comparator(pipe_embs[:, j], pipe_embs[:, j + 1]).squeeze(1) > 0.0
                else:
                    to_swap = comparator(pipe_embs[:, j], pipe_embs[:, j + 1], dataset_embeddings).squeeze(1) > 0.0
            swapped = to_swap.any().item()
            if not swapped:
                continue
            pipe_embs[to_swap, j], pipe_embs[to_swap, j + 1] = pipe_embs[to_swap, j + 1], pipe_embs[to_swap, j]
            indices[to_swap, j], indices[to_swap, j + 1] = indices[to_swap, j + 1], indices[to_swap, j]
        if not swapped:
            break
    return indices
