import pickle

from decompose.data_utils import load_standard_dataset
from decompose.dvc_utils import cwd_path
from decompose.utils import plot_decomp_grid, savefigs

import matplotlib.pyplot as plt
import numpy as np


def plot_margin_distr_by_trees(dataset_id, model_id, ax):
    decomp_path = cwd_path("decomps", dataset_id, model_id + ".pkl")
    with open(decomp_path, "rb") as f:
        decomp = pickle.load(f)
    margin_distr_by_trees(decomp, ax)

def margin_distr_by_trees(decomp, ax):
    correct_means = []
    correct_stds = []
    incorrect_means = []
    incorrect_stds = []
    ms = range(30)  # TODO move to params and use same value as in other plots
    for M in ms:
        e = decomp.error_function((), (0), M)
        t = e.mean(axis=0)
        correct_means.append(empty_save_mean(t[t < 0.5]))
        correct_stds.append(np.std(t[t < 0.5]))
        incorrect_means.append(empty_save_mean(t[t >= 0.5]))
        incorrect_stds.append(np.std(t[t >= 0.5]))
    correct_means = np.array(correct_means)
    correct_stds = np.array(correct_stds)
    incorrect_means = np.array(incorrect_means)
    incorrect_stds = np.array(incorrect_stds)
    ax.plot(ms, correct_means, color="green")
    ax.plot(ms, incorrect_means, color="red")
    ax.fill_between(ms, correct_means - correct_stds * 1 / 2, correct_means + correct_stds * 1 / 2, alpha=0.3,
                    facecolor="green")
    ax.fill_between(ms, incorrect_means - incorrect_stds * 1 / 2, incorrect_means + incorrect_stds * 1 / 2, alpha=0.3,
                    facecolor="red")


def empty_save_mean(arr):
    if len(arr) == 0:
        return 0
    return arr.mean()


if __name__ == "__main__":
    # mean margins for correct and incorrect examples over M

    gridfig, rowfigs, singlecell_figs = plot_decomp_grid(plot_margin_distr_by_trees)
    basepath = cwd_path("plots", "margins")
    kind = "margins"
    savefigs(basepath, kind, gridfig, rowfigs, singlecell_figs)
