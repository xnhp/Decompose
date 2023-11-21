from decompose.dvc_utils import cwd_path
from decompose.utils import plot_decomp_values, label, savefigs, load_saved_decomp
from decompose.plot_decomp_grid import plot_decomp_grid


def main():
    """
    Grid of plots with one row per dataset and one column per model, each plotting individual decomp terms.
    """

    import dvc.api
    params = dvc.api.params_show("params-getters.yaml")

    def consumer(dataset_id, model_id, ax):
        b = load_saved_decomp(dataset_id, model_id, "get_average_bias")
        v = load_saved_decomp(dataset_id, model_id, "get_average_variance_effect")
        m = b + v
        d = load_saved_decomp(dataset_id, model_id, "get_diversity_effect")
        ax.scatter(m[:, 1], d[:, 1], label="member error vs div-eff")
        # ax.plot(m[:,0], m[:,1], label="member error", color="green")
        # ax.plot(d[:,0], d[:,1], label="diversity effect", color="blue")

    binary_datasets = [
        'qsar-biodeg', "diabetes", "bioresponse", "spambase-openml"
    ]
    nonbinary_datasets = ['digits', 'mnist_subset', 'cover']

    all_datasets = binary_datasets + nonbinary_datasets

    tasks = [
        {
            'out_path': "plots/err_div/err_div",
            'datasets': binary_datasets,
            "models": [
                'standard_rf'
                # TODO maybe more
            ]
        },
    ]

    for task in tasks:
        gridfig, rowfigs, singlecell_figs = plot_decomp_grid(consumer, task)
        basepath = cwd_path(task['out_path'])
        kind = "bvd"
        savefigs(basepath, kind, gridfig, rowfigs, singlecell_figs)


if __name__ == "__main__":
    main()
