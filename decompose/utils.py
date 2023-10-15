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

        # "get_expected_member_loss_per_example": {"label": "\\frac{1}{M} \\sum_{i=1}^M \\mathbb{E} [L(y, q_i)]"},

        "get_average_bias": {"label": "$\\overline{bias}$"},
        "get_average_variance_effect": {"label": "$\\overline{var}$"},
        "get_diversity_effect": {"label": "div"},
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
    n_datasets = len(list(children(base_dir)))
    n_models = max([len(list(children(dataset_path))) for _, dataset_path in children(base_dir)])
    for dataset_info in enumerate(children(base_dir)):
        _, (_, dataset_path) = dataset_info
        for model_info in enumerate(children(dataset_path)):
            consumer(dataset_info, model_info)


def plot_data_model_grid(base_dir, plotter):
    plt.style.use(matplotx.styles.dufte)
    n_datasets = len(list(children(base_dir)))
    n_models = max([len(list(children(dataset_path))) for _, dataset_path in children(base_dir)])
    width = 12
    rowheight = 3
    fig, axs = plt.subplots(n_datasets, n_models + 1, figsize=(width, rowheight * n_datasets))
    for dataset_idx, (dataset_id, dataset_path) in enumerate(children(base_dir)):
        summary_ax = axs[dataset_idx, 0]
        plot_summary(dataset_id, summary_ax)
        for model_idx, (model_id, _) in enumerate(children(dataset_path)):
            if n_datasets == 1:
                axs[model_idx].set_title(f"{model_id}")
            else:
                axs[0, model_idx + 1].set_title(f"{model_id}")
            model_idx = model_idx + 1
            if n_datasets == 1:
                ax = axs[model_idx]
            else:
                ax = axs[dataset_idx, model_idx]
            ax.tick_params(axis='x', which='major', reset=True)
            ax.sharey(axs[dataset_idx, 1])  # share with first containing actual data

            # now actually plot
            plotter(dataset_id, model_id, ax)

    return fig


def plot_decomp_grid(getter_ids, target_filepath):
    plt.style.use(matplotx.styles.dufte)
    n_datasets = len(list(children(cwd_path("staged-decomp-values"))))
    n_models = max(
        [len(list(children(dataset_path))) for _, dataset_path in children(cwd_path("staged-decomp-values"))])
    # TODO spacing between rows
    width = 16
    rowheight = 3
    fig, axs = plt.subplots(n_datasets, n_models + 1, figsize=(width, rowheight * n_datasets))
    for dataset_idx, (dataset_id, dataset_path) in enumerate(children(cwd_path("staged-decomp-values"))):

        # axs[dataset_idx, 1].set_ylabel("train error")

        # summary_ax = axs[0] if n_datasets == 1 else axs[dataset_idx, 0]
        # TODO re-enable
        summary_ax = axs[dataset_idx, 0]
        plot_summary(dataset_id, summary_ax)

        for model_idx, (model_id, _) in enumerate(children(dataset_path)):

            if n_datasets == 1:
                axs[model_idx].set_title(f"{model_id}")
            else:
                axs[0, model_idx + 1].set_title(f"{model_id}")

            model_idx = model_idx + 1
            if n_datasets == 1:
                ax = axs[model_idx]
            else:
                ax = axs[dataset_idx, model_idx]

            ax.tick_params(axis='x', which='major', reset=True)
            ax.sharey(axs[dataset_idx, 1])  # share with first containing actual data

            for getter_id in getter_ids:
                plot_decomp_values(dataset_id, model_id, getter_id, ax, label=label(getter_id))

            matplotx.line_labels()  # line labels to the right
            # TODO shared legend

    # for ax in axs.flat:
    #     ax.label_outer()

    fig.tight_layout()
    fig.savefig(target_filepath)
