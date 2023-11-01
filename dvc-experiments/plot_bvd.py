from decompose.dvc_utils import cwd_path
from decompose.utils import plot_decomp_grid, plot_decomp_values, label, savefigs


def main():
    """
    Grid of plots with one row per dataset and one column per model, each plotting individual decomp terms.
    """

    import dvc.api
    params = dvc.api.params_show("params-getters.yaml")
    getters = params['plot_bvd_getters']

    def consumer(dataset_id, model_id, ax):
        for getter_id in getters:
            plot_decomp_values(dataset_id, model_id, getter_id, ax, label=label(getter_id))

    gridfig, rowfigs, singlecell_figs = plot_decomp_grid(consumer)

    basepath = cwd_path("plots", "bvd-decomps")
    kind = "bvd"

    savefigs(basepath, kind, gridfig, rowfigs, singlecell_figs)


if __name__ == "__main__":
    main()
