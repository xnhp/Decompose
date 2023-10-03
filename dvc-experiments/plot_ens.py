from decompose.dvc_utils import dataset_summary, cwd_path
from decompose.utils import plot_decomp_grid


def main():
    """
    Grid of plots with one row per dataset and one column per model, each plotting individual decomp terms.
    """

    plot_decomp_grid([
        "get_expected_ensemble_loss",
        "get_ensemble_bias",
        "get_ensemble_variance_effect"
    ],
    cwd_path("plots", "bvd-decomps", "ens.png")
    )

    # TODO this replaces staged errors
    # exp_ens_losses = [np.mean(decomp.get_expected_ensemble_loss(i)) for i in range(2,M)]
    # plt.plot(exp_ens_losses, label=decomp_id)

    # plt.legend()


if __name__ == "__main__":
    main()
