from decompose.dvc_utils import dataset_summary, cwd_path
from decompose.utils import plot_decomp_grid


def main():
    """
    Grid of plots with one row per dataset and one column per model, each plotting individual decomp terms.
    """

    import dvc.api
    params = dvc.api.params_show("params-getters.yaml")
    getters = params['plot_bvd_getters']

    plot_decomp_grid(
        getters,
        cwd_path("plots", "bvd-decomps", "bvd.png")
    )


if __name__ == "__main__":
    main()
