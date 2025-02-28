from decompose.dvc_utils import dataset_summary, cwd_path
from decompose.utils import plot_decomp_values, label, savefigs
from decompose.plot_decomp_grid import plot_decomp_grid


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

    binary_datasets = [
        'qsar-biodeg', "diabetes", "bioresponse", "spambase-openml"
    ]
    nonbinary_datasets = ['digits', 'mnist_subset', 'cover']

    all_datasets = binary_datasets + nonbinary_datasets

    tasks = [

        # "diversity is a measure of model fit"
        {
            'out_path': "plots/bvd-decomps/plot_bvd_standard_rf",
            'datasets': nonbinary_datasets,
            "models": [
                'standard_rf'
            ]
        }
    ]

    for task in tasks:
        gridfig, rowfigs, singlecell_figs = plot_decomp_grid(consumer, task)
        basepath = cwd_path(task['out_path'])
        kind = "ens"
        savefigs(basepath, kind, gridfig, rowfigs, singlecell_figs)

    # TODO this replaces staged errors
    # exp_ens_losses = [np.mean(decomp.get_expected_ensemble_loss(i)) for i in range(2,M)]
    # plt.plot(exp_ens_losses, label=decomp_id)

    # plt.legend()


if __name__ == "__main__":
    main()
