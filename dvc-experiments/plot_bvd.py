from decompose.dvc_utils import dataset_summary, cwd_path
from decompose.utils import plot_decomp_grid


def main():
    """
    Grid of plots with one row per dataset and one column per model, each plotting individual decomp terms.
    """

    plot_decomp_grid([
        "get_expected_ensemble_loss",
        "get_average_bias",
        "get_average_variance_effect",
        "get_diversity_effect"
    ],
    cwd_path("plots", "bvd-decomps", "bvd.png")
    )


if __name__ == "__main__":
    main()
