import matplotx
from matplotlib import pyplot as plt

from decompose import dvc_utils
from decompose.dvc_utils import cwd_path, get_model_color
from decompose.utils import children, load_saved_decomp, getters_and_labels, all_getters


def main():
    """
    Compare values of decomp terms (getters) across different models in the same plot.
    """
    plt.style.use(matplotx.styles.dufte)

    args = dvc_utils.parse_args()
    dataset_id = args.dataset

    import dvc.api
    params = dvc.api.params_show("params-getters.yaml")
    getters = set(params['plot_bvd_getters'] + params['plot_ens_getters'])
    getters_and_labels = [(g, all_getters()[g]['label']) for g in getters]

    dataset_path = cwd_path("staged-decomp-values", dataset_id)

    for getter_id, getter_label in getters_and_labels:

        mins = []

        for model_idx, (model_id, _) in enumerate(children(dataset_path)):
            plt.title(f"{dataset_id} / {getter_label}")
            x = load_saved_decomp(dataset_id, model_id, getter_id)
            if x is None:
                continue
            plt.plot(x[:, 0], x[:, 1], color=get_model_color(model_id), label=model_id)

            mins.append(
                (x[:, 1].min(), model_id)
            )

        s = ""
        for min, model_id in sorted(mins, key=lambda x: x[0]):
            s += f"{model_id}: {min:.3f}\n"
        plt.text(0.95, 0.05, s, transform=plt.gca().transAxes, fontsize=12, ha='right', va='bottom', linespacing=1.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(cwd_path("plots", "bvd-decomps", dataset_id, f"{getter_id}.png"))
        plt.close()

    pass


if __name__ == "__main__":
    main()