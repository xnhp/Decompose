import glob
import os
from enum import auto

import numpy as np
from strenum import StrEnum

from decompose.dvc_utils import cwd_path
from decompose.regressors import StandardRFRegressor, SquaredErrorGradientRFRegressor


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


def load_saved_decomp(dataset_id, model_id, getter_id):
    path = cwd_path("staged-decomp-values", dataset_id, model_id, f"{getter_id}.npy")
    return np.load(path)
    # for dataset_idx, dataset_path in enumerate(dataset_glob()):
    #     yield np.load(dataset_path + "/" + getter_id + ".npy")


def getters():
    return [
        "get_expected_ensemble_loss",
        "get_ensemble_bias",
        "get_average_bias",
        "get_average_variance_effect",
        "get_diversity_effect"
    ]


def getters_labels():
    return [
        "ens loss",
        "bias($\\bar{q}$)",
        "$\\overline{bias}$",
        "$\\overline{var}$",
        "div"
    ]


def children(base_path):
    for path in glob.glob(base_path + "/*"):
        basename = os.path.basename(path)
        basename = os.path.splitext(basename)[0]
        yield basename, path

def children_decomp_objs(dataset_path):
    for decomp_path in glob.glob(dataset_path + "/*.pkl"):
        decomp_id = os.path.basename(decomp_path)
        decomp_id = os.path.splitext(decomp_id)[0]
        yield decomp_id, decomp_path

