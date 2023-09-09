import glob
import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from decompose.dvc_utils import results_filepath_base, results_filepath, cwd_path
from decompose.experiments import load_results

def get_model_color(identifier: str):
    colors = {
        # regression
        'standard-rf-regressor': "blue",
        'sqerr-gradient-rf-regressor': "green",
        # classification
        'standard-rf-classifier': "blue",
        'drf-weighted-classifier': "red",
        'ensemble-weighted-classifier': "green"
    }
    return colors[identifier]

def main():
    # TODO xlim s.t. no padding left and right of line
    start_at = 10
    for subdir, dirs, files in os.walk(results_filepath_base()):
        for dir in dirs:
            dataset_id = dir

            for model_idx, filepath in enumerate(glob.iglob(f"{results_filepath_base()}/{dir}/*.pkl")):
                name = os.path.basename(filepath)
                model_id = os.path.splitext(name)[0]  # filename without suffix

                results = load_results(results_filepath(model_id, dataset_id))

                staged_errors = results.staged_errors
                error_rates = np.mean(staged_errors, axis=2)  # mean over training samples

                for trial_idx in range(error_rates.shape[0]):
                    # means, mins, maxs = self._aggregate_trials(error_rates[trial_idx])
                    rates = error_rates[trial_idx][start_at:]
                    if np.max(rates) > 1:
                        logging.warning("y values greater than 1, clipping")
                        # clipping has no effect otherwise
                    ys = np.clip(rates, 0, 1)
                    plt.plot(ys, color=get_model_color(model_id))
                    # https://stackoverflow.com/a/43069856/156884 but does not really make sense here
                    # plt.fill_between(xs, mins, maxs, alpha=0.3, facecolor="green")
                plt.plot([], [], label=model_id, color=get_model_color(model_id))

            plt.legend()
            # TODO always show first x-axis tick to make clear that we don't necessary start at zero
            plt.xlabel("Number of trees")
            plt.ylabel("Ensemble Risk")
            plt.title(dataset_id)
            # plt.show()
            plt.savefig(cwd_path("plots", "staged_errors", f"{dataset_id}.png"))
            plt.close()

    pass


if __name__ == "__main__":
    main()
