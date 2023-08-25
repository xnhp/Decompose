import numpy as np
import sklearn


def jaccard_distance(i, j):
    return 1 - sklearn.metrics.jaccard_score(i, j, zero_division=0.0)


def disagreement_distance(preds_i, preds_j):
    return np.average(preds_i != preds_j)


def dhat_distance(preds_i, preds_j):
    """
    Parameters
    ----------
    preds_i: predicted class labels of the i-th model
    preds_j: predicted class labels of the j-th model

    Returns
    -------
    \hatD_{card}

    Example
    -------
    preds_i = np.array([1,1,2,3,3])
    preds_j = np.array([1,1,4,3,5])
    print(dhat_distance(preds_i, preds_j)

    """
    # could be expressed even more succinctly by passing arrays of binary class indicators (i.e. avoiding the loop over k)
    #  https://scikit-learn.org/stable/modules/model_evaluation.html#jaccard-similarity-score
    # enough to consider max class idx and not n_classes because any class not appearing would contribute 0 to the sum
    n_classes = max(np.maximum.reduce([preds_i, preds_j]))
    return sum(jaccard_distance(preds_i == k, preds_j == k) for k in range(0, n_classes))


def discrete_metric(x, y):
    return 0 if x == y else 1

