from decompose.dvc_utils import dataset_summary, cwd_path
from decompose.utils import plot_decomp_grid, plot_decomp_values, label, savefigs


def main():
    """
    Grid of plots with one row per dataset and one column per model, each plotting individual decomp terms.
    """

    import dvc.api
    params = dvc.api.params_show("params-getters.yaml")
    getter_ids = params['plot_ens_getters']

    def consumer(dataset_id, model_id, ax):
        for getter_id in getter_ids:
            plot_decomp_values(dataset_id, model_id, getter_id, ax, label=label(getter_id))

    gridfig, rowfigs, singlecell_figs = plot_decomp_grid(consumer)

    basepath = cwd_path("plots", "bvd-decomps")
    kind = "ens"

    savefigs(basepath, kind, gridfig, rowfigs, singlecell_figs)

    # TODO this replaces staged errors
    # exp_ens_losses = [np.mean(decomp.get_expected_ensemble_loss(i)) for i in range(2,M)]
    # plt.plot(exp_ens_losses, label=decomp_id)

    # plt.legend()


if __name__ == "__main__":
    main()
