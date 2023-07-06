import os
from typing import Tuple

from cached_property import cached_property
from matplotlib import pyplot as plt
from sklearn import manifold

from decompose import BVDExperiment, plotting_utils
from decompose.experiments import load_results, AbstractResultsObject
from utils import CustomMetric

StandardDataset = Tuple


class MyExperiment(object):

    def __init__(self, identifier: str, dataset: StandardDataset, experiment: BVDExperiment, n_trials=1, debug=False,
                 custom_metrics=None):
        if custom_metrics is None:
            custom_metrics = set()
        self.debug = debug
        self.identifier = identifier
        self.results_filename = "results.pkl"
        self.experiment = experiment
        self.dataset = dataset
        self.n_trials = n_trials
        self.custom_metrics = custom_metrics

    @cached_property
    def get_results(self):

        print("accessing cached property")

        results_filepath = os.path.join(os.path.dirname(__file__), "experiments", self.identifier,
                                        self.results_filename)

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

    def plot(self, x_label=None, custom_metrics={}):
        ax = plotting_utils.plot_bvd(self.get_results, x_label=x_label, integer_x=True, custom_metrics=custom_metrics)
        fname = os.path.join(os.path.dirname(__file__), "experiments", self.identifier, "line-plot")
        plt.savefig(fname)
        plt.show()
        return ax

    def plot_mat_mds(self, cmetric, title):
        # to overlay multiple MDS plots, see https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html
        distmat = self.get_results[cmetric][(0, 0)]
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

        plt.title(f"MDS embedding of {title} pairw. distances ")
        # color names: https://www.w3schools.com/cssref/css_colors.php
        plt.scatter(pos[0, 0], pos[0, 1], color="CornflowerBlue", s=100, label="ground-truth")  # ground-truth
        plt.scatter(pos[1, 0], pos[1, 1], color="Brown", s=100, label="majority vote")  # central prediction (maj vote)
        plt.scatter(pos[2:, 0], pos[2:, 1], color="ForestGreen", s=100, lw=0, label="members")
        plt.legend()  # according to label params
        self._savefig(f"distmat_mds_{cmetric}")

    def plot_mat_heat(self, cmetric):
        mat = self.get_results[cmetric][(0, 0)]
        cax = plt.matshow(mat)
        plt.colorbar(cax)
        plt.title(cmetric)
        self._savefig(f"distmat_heat_{cmetric}")

    def _savefig(self, fname):
        savepath = os.path.join(os.path.dirname(__file__), "experiments", self.identifier, fname + ".png")
        plt.savefig(savepath)
        plt.close()
