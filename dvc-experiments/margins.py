import pickle

from decompose.data_utils import load_standard_dataset
from decompose.dvc_utils import cwd_path
from decompose.utils import load_saved_decomp, plot_data_model_grid, data_model_foreach

import matplotlib.pyplot as plt
import numpy as np


def main():

    # mean margins for correct and incorrect examples over M
    def plot_margin_distr_by_trees(dataset_id, model_id, ax):
        decomp_path = cwd_path("decomps", dataset_id, model_id + ".pkl")
        with open(decomp_path, "rb") as f:
            decomp = pickle.load(f)
        margin_distr_by_trees(decomp, ax)
    fig = plot_data_model_grid(cwd_path("decomps"), plot_margin_distr_by_trees)
    fig.savefig(cwd_path("plots", "margins", "margins-grid.png"))

    def plot_margin_distr_by_example(data_info, model_info):
        _, (dataset_id, _) = data_info
        _, (model_id, _) = model_info

        decomp_path = cwd_path("decomps", dataset_id, model_id + ".pkl")
        with open(decomp_path, "rb") as f:
            decomp = pickle.load(f)

        ms = [2, 5, 10, 20, 50, 100, 150]
        for m in ms:
            fig = mg_distr_by_example(m, decomp)
            fig.savefig(cwd_path("plots", "margins", dataset_id, model_id, f"by-example-{m}.png"))
    data_model_foreach(cwd_path("decomps"), plot_margin_distr_by_example)


def margin_distr_by_trees(decomp, ax):
    correct_means = []
    correct_stds = []
    incorrect_means = []
    incorrect_stds = []
    ms = range(30)
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
    ax.fill_between(ms, correct_means-correct_stds, correct_means+correct_stds, alpha=0.3, facecolor="green")
    ax.fill_between(ms, incorrect_means-incorrect_stds, incorrect_means+incorrect_stds, alpha=0.3, facecolor="red")


def mg_distr_by_example(M, decomp):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    e = decomp.error_function((), (0), M)
    mg_per_ex = e.mean(axis=0)
    below_half = mg_per_ex[mg_per_ex < 0.5]
    above_half = mg_per_ex[mg_per_ex >= 0.5]
    ax.hist(below_half, bins=100, alpha=0.5, label="below half", color="green")
    ax.hist(above_half, bins=100, alpha=0.5, label="above half", color="red")
    ax.axvline(1 / 2, color="black", label="0.5", alpha=0.5)
    ax.set_xlabel("Margin (ratio incorrect trees)")
    ax.set_title(f"margin distr. by example for M={M}")
    return fig


def empty_save_mean(arr):
    if len(arr) == 0:
        return 0
    return arr.mean()


if __name__ == "__main__":
    main()