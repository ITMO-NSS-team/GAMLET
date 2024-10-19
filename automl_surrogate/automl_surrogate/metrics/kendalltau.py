import numpy as np
from scipy.stats import kendalltau as kendalltau_
from torch import Tensor


def kendalltau(true_scores: Tensor, pred_scores: Tensor, top_k: int = None) -> float:
    res = [kendalltau_(true, pred).correlation for true, pred in zip(true_scores, pred_scores)]
    return np.mean(res).item()
