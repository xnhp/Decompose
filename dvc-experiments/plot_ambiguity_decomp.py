"""
Similar to Fig20 of wood23
"""
import glob
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from decompose.dvc_utils import cwd_path, get_n_classes


def main():

    for dataset_path in glob.glob(cwd_path("decomps") + "/*"):
        dataset_id = os.path.basename(dataset_path)

        n_classes = get_n_classes(dataset_id)

        for decomp_path in glob.glob(dataset_path + "/*.pkl"):
            with open(decomp_path, "rb") as f:
                decomp = pickle.load(f)

            pred = decomp.pred # [trials, estimators, examples]
            labels = decomp.labels  # 1D in case of ZeroOneLoss [examples]
            # if I understand correctly, diversity-effect is E_D, i.e. over trials
            # so, averaging over trials below as well
            models_correct = (pred == labels).mean(axis=(0,1))
            plt.scatter(
                1 - models_correct,
                decomp.diversity_effect,
                c=labels,
                cmap="Set3",  # needs to have enough values
                label="examples in test set"
            )

            plt.axhline(0, color="gray", alpha = 0.4)

            if n_classes is not None:
                random_guesser_rate = 1 - 1/n_classes
                plt.axvline(random_guesser_rate, color="red", label="$(1 - $n_classes$^{-1})$: random guesser error rate", alpha=0.5)
                x = np.linspace(0, random_guesser_rate, 1000)
                div_bounds_params = {
                    "color": "cyan",
                    "alpha": 0.4,
                    # "label": "div-eff bounds??"
                }
                plt.plot(x, x, **div_bounds_params)
                x = np.linspace(random_guesser_rate, 1, 1000)
                plt.plot(x, -(1-x), **div_bounds_params)

            decomp_id = os.path.basename(decomp_path)
            decomp_id = os.path.splitext(decomp_id)[0]
            plt.title(f"{decomp_id} on {dataset_id}")
            plt.savefig(cwd_path("plots", "ambiguity_decomp", f"{dataset_id}", f"{decomp_id}.png"))
            plt.ylabel("Diversity-Effect")
            plt.xlabel("Average individual error")
            plt.legend()
            # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot/43439132#43439132
            plt.tight_layout()
            plt.close()

    # for subdir, dirs, files in os.walk(cwd_path("decomps")):
    #     for dataset_id in dirs:
    #         path = f"{results_filepath_base()}/{dataset_id}/*.pkl"
    #         globbed = glob.iglob(path)
    #         for model_idx, filepath in enumerate(globbed):
    #             name = os.path.basename(filepath)
    #             model_id = os.path.splitext(name)[0]  # filename without suffix
    #             results = load_results(results_filepath(model_id, dataset_id))
    #             print("hello")
    #
    #
    #         plt.legend()
    #         # TODO always show first x-axis tick to make clear that we don't necessary start at zero
    #         plt.xlabel("Number of trees")
    #         plt.ylabel("Ensemble Risk")
    #         plt.title(dataset_id)
    #         # plt.show()
    #         plt.savefig(cwd_path("plots", "staged_errors", f"{dataset_id}.png"))
    #         plt.close()


if __name__ == "__main__":
    main()