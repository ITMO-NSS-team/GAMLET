import torch
from torch import Tensor


def precision(true_scores: Tensor, pred_scores: Tensor, top_k: int = None) -> float:
    """Accepts pipelines scores. Then sort in descending order. Compute metric per sample and then average."""
    res = torch.argsort(true_scores, dim=1, descending=True) == torch.argsort(pred_scores, dim=1, descending=True)
    if top_k is not None:
        res = res[:, :top_k]
    res = res.to(torch.float32)
    return res.mean().item()
