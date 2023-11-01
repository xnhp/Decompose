import glob
import logging
import os
from enum import auto

import matplotx
import numpy as np
from matplotlib import pyplot as plt
from strenum import StrEnum

from decompose.dvc_utils import cwd_path, get_fn_color, dataset_summary
from decompose.regressors import StandardRFRegressor, SqErrBoostedBase


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

def plot_decomp_grid(consumer):
    plt.style.use(matplotx.styles.dufte)
    n_datasets = len(list(children(cwd_path("staged-decomp-values"))))
    n_models = max(
        [len(list(children(dataset_path))) for _, dataset_path in children(cwd_path("staged-decomp-values"))])
    # TODO spacing between rows
    colwidth = 24
    rowheight = 3
    n_rows = n_datasets
    n_cols = n_models + 1  # +1 for dataset summary
    gridfig, gridaxs = plt.subplots(n_rows, n_cols, figsize=(colwidth, rowheight * n_rows))

    col_indices = reverse(dict(enumerate([
        'standard-rf-classifier',
        'drf-weighted-bootstrap-classifier',
        'sigmoid-weighted-bootstrap-classifier',
        'xuchen-weighted-bootstrap-classifier',
        'drf-weighted-fit-classifier',
        'drf-weighted-fit-oob-classifier'
    ])))

    dataset_indices = reverse(dict(enumerate([
        # large to small
        'cover',
        'mnist_subset',
        'spambase-openml',
        'bioresponse',
        'digits',
        'diabetes',
        'qsar-biodeg',
    ])))

    rowfigs = []
    singlecell_figs = []

    for dataset_idx, (dataset_id, dataset_path) in enumerate(children(cwd_path("staged-decomp-values"))):

        # also save row in separate plot
        rowfig, rowfigaxs = plt.subplots(1, n_cols, figsize=(colwidth, 4))

        plot_summary(dataset_id, rowfigaxs[0])
        row_index = dataset_indices[dataset_id]
        plot_summary(dataset_id, gridaxs[row_index, 0])

        model_results = children(dataset_path)
        for _, (model_id, _) in enumerate(model_results):

            # save each cell to a separate plot

            col_index = col_indices[model_id] + 1

            gridaxs[0, col_index].set_title(f"{model_id}")
            rowfigaxs[col_index].set_title(f"{model_id}")

            singlecell_fig, singlecell_ax = plt.subplots(1,1, figsize=(4,4))
            gridcell_ax = gridaxs[row_index, col_index]

            target_axs = [gridcell_ax, rowfigaxs[col_index], singlecell_ax]

            def set_tick_params(ax):
                ax.tick_params(axis='x', which='major', reset=True)
            map(set_tick_params, target_axs)

            gridcell_ax.sharey(gridaxs[row_index, 1])  # share with first containing actual data
            rowfigaxs[col_index].sharey(rowfigaxs[1])

            for ax in target_axs:
                consumer(dataset_id, model_id, ax)
            # for getter_id in getter_ids:
            #     plot_decomp_values(dataset_id, model_id, getter_id, gridcell_ax, label=label(getter_id))
            #     plot_decomp_values(dataset_id, model_id, getter_id, singlecell_ax, label=label(getter_id))
            #     plot_decomp_values(dataset_id, model_id, getter_id, rowfigaxs[col_index], label=label(getter_id))

            # TODO get legend right
            # matplotx.line_labels()  # line labels to the right

            singlecell_figs.append((dataset_id, model_id, singlecell_fig))
            # singlecell_fig.tight_layout()
            # singlecell_fig.savefig(cwd_path(target_dir, dataset_id, "bvd-individual", f"{model_id}.png"))

        rowfigs.append((dataset_id, rowfig))
        # rowfig.tight_layout()
        # rowfig.savefig(cwd_path(target_dir, dataset_id, f"{kind}.png"))

    # gridfig.tight_layout()
    # gridfig.savefig(cwd_path(target_dir, f"{kind}.png"))

    return gridfig, rowfigs, singlecell_figs


def savefigs(basepath, kind, gridfig, rowfigs, singlecell_figs):
    gridfig.tight_layout()
    gridfig.savefig(cwd_path(basepath, f"{kind}.png"))
    for dataset_id, rowfig in rowfigs:
        rowfig.tight_layout()
        rowfig.savefig(cwd_path(basepath, dataset_id, f"{kind}.png"))
    for dataset_id, model_id, singlecell_fig in singlecell_figs:
        singlecell_fig.tight_layout()
        singlecell_fig.savefig(cwd_path(basepath, dataset_id, f"{model_id}.png"))
