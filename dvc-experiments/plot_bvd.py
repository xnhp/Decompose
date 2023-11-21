from decompose.dvc_utils import cwd_path
from decompose.utils import plot_decomp_values, label, savefigs
from decompose.plot_decomp_grid import plot_decomp_grid


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

    binary_datasets = [
        'qsar-biodeg', "diabetes", "bioresponse", "spambase-openml"
    ]
    nonbinary_datasets = ['digits', 'mnist_subset', 'cover']

    all_datasets = binary_datasets + nonbinary_datasets

    tasks = [

        # "diversity is a measure of model fit"
        # cf corresponding in plot_ens
        {
            'out_path': "plots/bvd-decomps/plot_bvd_standard_rf",
            'datasets': nonbinary_datasets,
            "models": [
                'standard_rf'
            ]
        },

        {
            'out_path': "plots/bvd-decomps/plot_bvd_drf",
            'datasets': binary_datasets,
            "models": [
                'standard_rf',
                'drf_weighted_bootstrap',
                'drf_weighted_fit',
                'xu_chen',
            ]
        },

        {
            'out_path': "plots/bvd-decomps/plot_bvd_capped_lerped_sigmoid",
            'datasets': binary_datasets,
            "models": [
                'standard_rf',
                "drf_weighted_bootstrap",
                "capped_sigmoid",
                "capped_lerped_sigmoid"
            ]
        },

        {
            'out_path': "plots/bvd-decomps/plot_bvd_multiclass",
            'datasets': all_datasets,
            "models": [
                'standard_rf',
                # "drf_weighted_bootstrap",
                "capped_lerped_sigmoid",
                "dynamic_threshold"
            ]
        }
    ]

    for task in tasks:
        gridfig, rowfigs, singlecell_figs = plot_decomp_grid(consumer, task)
        basepath = cwd_path(task['out_path'])
        kind = "bvd"
        savefigs(basepath, kind, gridfig, rowfigs, singlecell_figs)


if __name__ == "__main__":
    main()
