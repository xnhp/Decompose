import logging
import os
from typing import Tuple

import numpy as np
from cached_property import cached_property
from matplotlib import pyplot as plt
from sklearn import manifold

from decompose import BVDExperiment, plotting_utils
from decompose.experiments import load_results, AbstractResultsObject
from utils import CustomMetric

StandardDataset = Tuple


class MyExperiment(object):

    def __init__(self, file:str, dataset: StandardDataset, experiment: BVDExperiment, n_trials=1, debug=False,
                 custom_metrics=None):
        if custom_metrics is None:
            custom_metrics = set()
        self.debug = debug
        self.path = os.path.dirname(file)  # directory where results of this experiment are expected
        self.identifier = os.path.basename(os.path.dirname(file))
        self.results_filename = "results.pkl"
        self.experiment = experiment
        self.dataset = dataset
        self.n_trials = n_trials
        self.custom_metrics = custom_metrics

    @cached_property
    def get_results(self):

        print("accessing cached property")

        results_filepath = os.path.join(self.path, self.results_filename)

        if os.path.exists(results_filepath) and not self.debug:
            print("Saved results present, using these (possibly out of sync, check git status)")
            results = load_results(results_filepath)
        else:
            print("Computing results")
            results = self.experiment.run_experiment(*self.dataset, n_trials=self.n_trials,
                                                     custom_metrics=self.custom_metrics)
            if not self.debug:
                print("saving results")
                results.save_results(results_filepath)
            else:
                print("not saving results because debug=True")

        return results

    def plot(self, x_label=None, custom_metrics={}, show=False):
        ax = plotting_utils.plot_bvd(
            self.get_results,
            x_label=x_label,
            integer_x=True,
            custom_metrics=custom_metrics,
            ensemble_bias=CustomMetric.ENSEMBLE_BIAS in custom_metrics,
            ensemble_variance=CustomMetric.ENSEMBLE_VARIANCE in custom_metrics
        )
        fname = os.path.join(self.path, "line-plot")
        plt.savefig(fname)
        if show:
            plt.show()
        return ax

    def _plot_mat_mds(self, distmat, cmetric, param_idx, title):
        # to overlay multiple MDS plots, see https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html
        mds = manifold.MDS(
            n_components=2,
            max_iter=3000,
            eps=1e-9,
            random_state=0,
            dissimilarity="precomputed",
            n_jobs=4,
            n_init=4
        )
        mds_fit = mds.fit(distmat)
        pos = mds_fit.embedding_

        print(f"stress: {mds_fit.stress_}, n_iter_: {mds_fit}")

        parameter_name = self.experiment.parameter_name
        parameter_value = self.experiment.parameter_values[param_idx]
        plt.title(f"MDS embedding of {title} pairw. distances at {parameter_name} {parameter_value}  ")
        # color names: https://www.w3schools.com/cssref/css_colors.php
        plt.scatter(pos[0, 0], pos[0, 1], color="CornflowerBlue", s=100, label="ground-truth")  # ground-truth
        plt.scatter(pos[1, 0], pos[1, 1], color="Brown", s=100, label="majority vote")  # central prediction (maj vote)
        plt.scatter(pos[2:, 0], pos[2:, 1], color="ForestGreen", s=100, lw=0, label="members")
        plt.legend()  # according to label params
        self._savefig(f"distmat_mds-{cmetric}-{parameter_name}-{parameter_value}")

    def plot_mat_mds(self, cmetric, title):
        for param_idx, split_idx in self.get_results[cmetric]:
            self._plot_mat_mds(self.get_results[cmetric][(param_idx, split_idx)], cmetric, param_idx, title)

    def plot_mat_heat(self, cmetric):
        for param_idx, split_idx in self.get_results[cmetric]:
            parameter_name = self.experiment.parameter_name
            parameter_value = self.experiment.parameter_values[param_idx]
            mat = self.get_results[cmetric][(param_idx, split_idx)]
            cax = plt.matshow(mat)
            plt.colorbar(cax)
            plt.title(cmetric)
            self._savefig(f"distmat_heat-{cmetric}-{parameter_name}-{parameter_value}")

    def _savefig(self, fname):
        savepath = os.path.join(self.path, fname + ".png")
        plt.savefig(savepath)
        plt.close()


def plot_staged_errors(experiments, attr_getter, title=None, start_at=0):
    colors = ["green", "orange", "blue", "red", "yellow", "brown"]
    for exp_idx, exp in enumerate(experiments):
        errs = attr_getter(exp)
        error_rates = np.mean(errs, axis=2)
        for trial_idx in range(error_rates.shape[0]):
            # means, mins, maxs = self._aggregate_trials(error_rates[trial_idx])
            rates = error_rates[trial_idx][start_at:]
            if np.max(rates) > 1:
                logging.warning("y values greater than 1, clipping")
                # clipping has no effect otherwise
            ys = np.clip(rates, 0, 1)
            plt.plot(ys, label=exp.identifier, color=colors[exp_idx])
            # https://stackoverflow.com/a/43069856/156884 but does not really make sense here
            # plt.fill_between(xs, mins, maxs, alpha=0.3, facecolor="green")
    plt.legend()
    # TODO always show first x-axis tick to make clear that we don't necessary start at zero
    plt.xlabel("Number of trees")
    plt.ylabel("Ensemble Risk")
    plt.title(title)
    plt.show()

