import glob
import logging
import os
from enum import auto

import numpy as np
from strenum import StrEnum

from decompose.dvc_utils import cwd_path, get_fn_color, dataset_summary


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
    try:
        return np.load(path)
    except FileNotFoundError as e:
        logging.error(f"Error loading {path}: {e}")
        return None
    # for dataset_idx, dataset_path in enumerate(dataset_glob()):
    #     yield np.load(dataset_path + "/" + getter_id + ".npy")


def all_getters():
    return {
        "get_expected_ensemble_loss": {"label": "ens loss"},

        "get_ensemble_bias": {"label": "bias($\\bar{q}$)"},
        "get_ensemble_variance_effect": {"label": "var($\\bar{q}$)"},
        "get_ensemble_variance": {"label": "var($\\bar{q}$)"},

        # "get_expected_member_loss_per_example": {"label": "\\frac{1}{M} \\sum_{i=1}^M \\mathbb{E} [L(y, q_i)]"},

        "get_average_bias": {"label": "$\\overline{bias}$"},
        "get_average_variance_effect": {"label": "$\\overline{var}$"},
        "get_average_variance": {"label": "$\\overline{var}$"},
        "get_diversity_effect": {"label": "div"},
        "get_diversity": {"label": "div"},
    }

def getters_and_labels():
    return [(id, all_getters()[id]['label']) for id in all_getters().keys()]

def label(id):
    return all_getters()[id]['label']

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


def plot_summary(dataset_id, summary_axs):
    summary = dataset_summary(dataset_id)
    summary_text = f"""
        {dataset_id}
        {summary["n_classes"]} classes
        {summary["n_train"]} train samples
        {summary["n_test"]} test samples
        {summary["dimensions"]} features
        """
    # TODO
    summary_axs.text(0.1, 0.5, summary_text, fontsize=12, ha='left', va='center', linespacing=1.5, transform=summary_axs.transAxes, color="black")
    summary_axs.set_xticks([])
    summary_axs.set_yticks([])

def plot_decomp_values(dataset_id, model_id, getter_id, ax, label=None):
    x = load_saved_decomp(dataset_id, model_id, getter_id)
    if x is None:
        return
    ax.plot(x[:, 0], x[:, 1], color=get_fn_color(getter_id), label=label)


def data_model_foreach(base_dir, consumer):
    # TODO same order as in other grids
    for dataset_info in enumerate(children(base_dir)):
        _, (_, dataset_path) = dataset_info
        for model_info in enumerate(children(dataset_path)):
            consumer(dataset_info, model_info)


def reverse(dict):
    inv_map = {v: k for k, v in dict.items()}
    return inv_map


def savefigs(basepath, kind, gridfig, rowfigs, singlecell_figs):
    gridfig.tight_layout()
    gridfig.savefig(cwd_path(basepath, f"{kind}.png"))
    for dataset_id, rowfig in rowfigs:
        rowfig.tight_layout()
        rowfig.savefig(cwd_path(basepath, dataset_id, f"{kind}.png"))
    for dataset_id, model_id, singlecell_fig in singlecell_figs:
        singlecell_fig.suptitle(f"{model_id}")
        singlecell_fig.tight_layout()
        singlecell_fig.savefig(cwd_path(basepath, dataset_id, f"{kind}-{model_id}.png"))
