from enum import auto

import numpy as np
from strenum import StrEnum

from decompose.regressors import StandardRFRegressor


def get_model(identifier: str):
    models = {
        'standard-rf-regressor': StandardRFRegressor()
    }
    return models[identifier]

def pairwise_matrix(data, fun):
    """

    Parameters
    ----------
    data            1D-Array of points
    fun     Compare two points

    Returns
    -------
    Symmetric matrix of fun(data[i], data[j])
    """
    m = data.shape[0]
    combinations_indices = np.transpose(np.triu_indices(m, 1))
    r = np.zeros([m, m])
    for i, j in combinations_indices:
        d_ij = fun(data[i].squeeze(), data[j].squeeze())
        r[i, j] = d_ij
        r[j, i] = d_ij
    return r


class CustomMetric(StrEnum):
    MEMBER_DEVIATION = auto()
    DISTMAT_DHAT = auto()
    DISTMAT_DISAGREEMENT = auto()
    INDIV_ERRORS = auto()
    EXP_MEMBER_LOSS = auto()
    COVMAT = auto()
    ENSEMBLE_BIAS = auto()
    ENSEMBLE_VARIANCE = auto()

def metric_display_name(cmetric: CustomMetric):
    names = {
        CustomMetric.MEMBER_DEVIATION: "member deviation",
    }
    if cmetric in names:
        return names[cmetric]
    else:
        print("no display name assigned for " + cmetric)
        return cmetric

class DatasetId(StrEnum):
    MNIST = "mnist"
    WINE = "wine"
