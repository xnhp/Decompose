import matplotx
from matplotlib import pyplot as plt

from decompose import dvc_utils
from decompose.dvc_utils import cwd_path, get_model_color, get_model_latex
from decompose.utils import children, load_saved_decomp, getters_and_labels, all_getters


def main():
    """
    Compare values of decomp terms (getters) across different models in the same plot.
    """
    plt.style.use(matplotx.styles.dufte)

    import dvc.api
    params = dvc.api.params_show("params-getters.yaml")
    getters = set(params['plot_bvd_getters'] + params['plot_ens_getters'])
    getters_and_labels = [(g, all_getters()[g]['label']) for g in getters]


    binary_datasets = [
        'qsar-biodeg', "diabetes", "bioresponse", "spambase-openml"
    ]
    nonbinary_datasets = ['digits', 'mnist_subset', 'cover']

    all_datasets = binary_datasets + nonbinary_datasets

    tasks = [
        {
            'id': 'drf_sigmoid',
            'datasets': all_datasets,
            "models": [
                'standard_rf',
                'drf_weighted_bootstrap',
                'capped_sigmoid',
                'capped_lerped_sigmoid'
            ]
        },
        {
            'id': 'dynamic_threshold',
            'datasets': nonbinary_datasets,
            "models": [
                'standard_rf',
                'drf_weighted_bootstrap',
                'capped_lerped_sigmoid',
                'dynamic_threshold'
            ]
        }
    ]

    for task in tasks:

        mins = {}
        for dataset_id in task['datasets']:
            for getter_id, getter_label in getters_and_labels:
                model_mins = {}
                for model_id in task['models']:

                    plt.title(f"{dataset_id} / {getter_label}")
                    x = load_saved_decomp(dataset_id, model_id, getter_id)
                    if x is None:
                        continue
                    plt.yscale("log")
                    plt.plot(x[:, 0], x[:, 1], color=get_model_color(model_id), label=model_id)

                    vals = x[:, 1]
                    min_index = vals.argmin()
                    min_value = vals[min_index]
                    model_mins[model_id] = (min_value, min_index)

                # s = ""
                # for min, model_id in sorted(mins, key=lambda x: x[0]):
                #     s += f"{model_id}: {min:.3f}\n"
                # plt.text(0.95, 0.05, s, transform=plt.gca().transAxes, fontsize=12, ha='right', va='bottom', linespacing=1.5)
                if getter_id == "get_expected_ensemble_loss":
                    mins[dataset_id] = model_mins

                plt.legend()
                plt.tight_layout()
                plt.savefig(cwd_path("plots", "compare_models", task['id'], dataset_id, f"{getter_id}.png"))
                plt.close()

        out_s = ""
        out_s += ","
        for model_id in task['models']:
            model_id_pretty = get_model_latex(model_id)
            out_s += "" + model_id_pretty + ","
        out_s += '\n'
        for dataset_id in mins:
            model_mins = mins[dataset_id]
            dataset_id_pretty = dataset_id.replace("_", "-")
            out_s += dataset_id_pretty + ","
            minvalue = min([m for m, _ in model_mins.values()])
            for mini, index in model_mins.values():
                if mini == minvalue:
                    out_s += f"$\\mathbf{{{mini:.3f}}}$"
                else:
                    out_s += f"{mini:.3f}"
                # could do this but minimum over all is not suited to answer question about smaller ensembes
                # out_s += f" (${index}$)"
                out_s += ","
            out_s += '\n'
        with open(cwd_path("plots", "compare_models", task['id'], f"get_expected_ensemble_loss.csv"), "w+", encoding="utf-8") as f:
            f.write(out_s)



if __name__ == "__main__":
    main()