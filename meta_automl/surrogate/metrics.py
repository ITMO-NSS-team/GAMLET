import warnings

import numpy as np
from scipy.stats import rankdata
import math
from typing import List
from sklearn.metrics import ndcg_score


def ndcg_fixed(y_true: list, y_pred: list, k=10) -> float:
    ndcg_score_skl = []
    k_lim = 3
    for idx in range(len(y_true)):
        tr = np.zeros((y_true[idx].shape[0],))
        tr[:k_lim] = 1
        ndcg_score_skl.append(ndcg_score([tr], [y_pred[idx]], k=k_lim))
    return np.mean(ndcg_score_skl)


def _precision(predicted, actual):
    prec = [value for value in predicted if value in actual]
    prec = float(len(prec)) / float(len(predicted))
    return prec


def _apk(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average precision at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average precision at k.
    """

    if len(predicted) > k:
        predicted = predicted[:k]

    predicted = np.argsort(predicted)[::-1]
    actual = np.argsort(actual)[::-1]

    print(predicted, actual)

    score = 0.0
    true_positives = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            max_ix = min(i + 1, len(predicted))
            score += _precision(predicted[:max_ix], actual)
            true_positives += 1

    if score == 0.0:
        return 0.0
    print(score, true_positives)

    return score / true_positives


def mapk(actual: List[list], predicted: List[list], k: int = 10) -> float:
    """
    Computes the mean average precision at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: float
            The mean average precision at k (map@k)
    """
    if len(actual) != len(predicted):
        raise AssertionError("Length mismatched")
    return np.mean([_apk(a, p, k) for a, p in zip(actual, predicted)])


def eval(y_pos, y_neg):
    hits1_list = []
    hits3_list = []
    hits7_list = []
    mrr_list = []
    for y_pred_neg, y_pred_pos in zip(y_neg, y_pos):
        y_pred = np.concatenate([y_pred_pos, y_pred_neg])
        ranking_list = (len(y_pred) - y_pred.argsort().argsort())
        hits1_list.append(int(any(ranking_list[:1] <= 1)))
        hits3_list.append(int(any(ranking_list[:3] <= 3)))
        hits7_list.append(int(any(ranking_list[:7] <= 7)))
        mrr_list.append(np.mean(1. / (np.where(ranking_list == 1)[0] + 1)))
    return np.mean(mrr_list), np.mean(hits1_list), np.mean(hits3_list), np.mean(hits7_list)


def dcg(order):
    log = 0
    for i, o in enumerate(order):
        log += o / math.log(1 + i + 1, 2)
    return np.array(log)


def ndcg(y_true_list, y_score_list):
    _ndcg = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        rank = rankdata(y_true, method='min')
        rank_true = rankdata(y_true[(-1 * rank).argsort()], method='max') - 1
        rank_pred = rankdata(y_score[(-1 * rank).argsort()], method='max') - 1

        idcg = np.mean(dcg(rank_true))
        _dcg = np.mean(dcg(rank_pred))

        _ndcg.append(_dcg / idcg)

    return np.mean(_ndcg)

###########################


def _mean_ranking_metric(predictions, labels, k, metric):
    """Helper function for precision_at_k and mean_average_precision"""
    result = []
    for i, prd in enumerate(predictions):
        # prd = np.argsort(prd)[::-1]
        # lbs = np.argsort(labels[i])[::-1]
        lbs = labels[i]
        # print(prd, lbs)
        result.append(metric(np.asarray(prd), np.asarray(lbs), k))
        # print(result[i])
    return np.mean(result)


def _warn_for_empty_labels():
    """Helper for missing ground truth sets"""
    warnings.warn("Empty ground truth set! Check input data")
    return 0.0


def precision_at(predictions, labels, k=10, assume_unique=True):
    """Compute the precision at K.

    Compute the average precision of all the queries, truncated at
    ranking position k. If for a query, the ranking algorithm returns
    n (n is less than k) results, the precision value will be computed
    as #(relevant items retrieved) / k. This formula also applies when
    the size of the ground truth set is less than k.

    If a query has an empty ground truth set, zero will be used as
    precision together with a warning.

    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.

    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).

    k : int, optional (default=10)
        The rank at which to measure the precision.

    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.

    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> precision_at(preds, labels, 1)
    0.33333333333333331
    >>> precision_at(preds, labels, 5)
    0.26666666666666666
    >>> precision_at(preds, labels, 15)
    0.17777777777777778
    """
    # validate K
    _require_positive_k(k)

    def _inner_pk(pred, lab, k=10):
        # need to compute the count of the number of values in the predictions
        # that are present in the labels. We'll use numpy in1d for this (set
        # intersection in O(1))
        if lab.shape[0] > 0:
            n = min(pred.shape[0], k)
            cnt = np.in1d(pred[:n], lab, assume_unique=assume_unique).sum()
            return float(cnt) / k
        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, k, _inner_pk)


def mean_average_precision_at(predictions, labels, k=10, assume_unique=True):
    """Compute the mean average precision on predictions and labels.

    Returns the mean average precision (MAP) of all the queries. If a query
    has an empty ground truth set, the average precision will be zero and a
    warning is generated.

    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.

    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).

    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.

    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>>mean_average_precision_at(preds, labels)
    0.35502645502645497
    >>>mean_average_precision_at(preds, labels, k=10)
    0.35502645502645497
    >>>mean_average_precision_at(preds, labels, k=3)
    0.2407407407407407
    >>>mean_average_precision_at(preds, labels, k=5)
    0.2111111111111111
    """

    def _inner_map(pred, lab, k=10):
        if lab.shape[0]:
            # compute the number of elements within the predictions that are
            # present in the actual labels, and get the cumulative sum weighted
            # by the index of the ranking
            n = min(pred.shape[0], k)

            arange = np.arange(n, dtype=np.float32) + 1.0  # this is the denom
            present = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            prec_sum = np.ones(present.sum()).cumsum()
            denom = arange[present]
            return (prec_sum / denom).sum() / min(lab.shape[0], k)

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, k, _inner_map)


def _require_positive_k(k):
    """Helper function to avoid copy/pasted code for validating K"""
    if k <= 0:
        raise ValueError("ranking position k should be positive")


def ndcg_at(predictions, labels, k=10, assume_unique=True):
    """Compute the normalized discounted cumulative gain at K.

    Compute the average NDCG value of all the queries, truncated at ranking
    position k. The discounted cumulative gain at position k is computed as:

        sum,,i=1,,^k^ (2^{relevance of ''i''th item}^ - 1) / log(i + 1)

    and the NDCG is obtained by dividing the DCG value on the ground truth set.
    In the current implementation, the relevance value is binary.

    If a query has an empty ground truth set, zero will be used as
    NDCG together with a warning.

    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.

    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).

    k : int, optional (default=10)
        The rank at which to measure the NDCG.

    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.

    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> ndcg_at(preds, labels, 3)
    0.3333333432674408
    >>> ndcg_at(preds, labels, 10)
    0.48791273434956867

    References
    ----------
    .. [1] K. Jarvelin and J. Kekalainen, "IR evaluation methods for
           retrieving highly relevant documents."
    """
    # validate K
    _require_positive_k(k)

    def _inner_ndcg(pred, lab, k=10):
        if lab.shape[0]:
            # if we do NOT assume uniqueness, the set is a bit different here
            if not assume_unique:
                lab = np.unique(lab)

            n_lab = lab.shape[0]
            n_pred = pred.shape[0]
            n = min(max(n_pred, n_lab), k)  # min(min(p, l), k)?

            # similar to mean_avg_prcsn, we need an arange, but this time +2
            # since python is zero-indexed, and the denom typically needs +1.
            # Also need the log base2...
            arange = np.arange(n, dtype=np.float32)  # length n

            # since we are only interested in the arange up to n_pred, truncate
            # if necessary
            arange = arange[:n_pred]
            denom = np.log2(arange + 2.0)  # length n
            gains = 1.0 / denom  # length n

            # compute the gains where the prediction is present in the labels
            dcg_mask = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            dcg = gains[dcg_mask].sum()

            # the max DCG is sum of gains where the index < the label set size
            max_dcg = gains[arange < n_lab].sum()
            return dcg / max_dcg

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, k, _inner_ndcg)
