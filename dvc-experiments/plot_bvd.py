import glob

import matplotx
from matplotlib import pyplot as plt

from decompose.dvc_utils import cwd_path, get_fn_color, \
    dataset_summary
from decompose.utils import load_saved_decomp, getters, getters_labels, children, children_decomp_objs


def dataset_glob():
    return glob.glob(cwd_path("decomps") + "/*")


def decomp_glob(dataset_path):
    return glob.glob(dataset_path + "/*.pkl")

def plot_decomp_values(dataset_id, model_id, getter_id, ax, label=None):
    x = load_saved_decomp(dataset_id, model_id, getter_id)
    ax.plot(x[:, 0], x[:, 1], color=get_fn_color(getter_id), label=label)

def plot_summary(dataset_id, summary_axs):
    summary = dataset_summary(dataset_id)
    summary_text = f"""
        {dataset_id}
        {summary["n_classes"]} classes
        {summary["n_train"]} train samples
        {summary["n_test"]} test samples
        {summary["dimensions"]} features
        """
    # TODO
    # summary_axs.text(0.1, 0.5, "foo", fontsize=12, ha='left', va='center', linespacing=1.5)
    # summary_axs.set_xticks([])
    # summary_axs.set_yticks([])
    summary_axs.set_ylabel(dataset_id)


def main():
    """
    Grid of plots with one row per dataset and one column per model, each plotting individual decomp terms.
    """
    plt.style.use(matplotx.styles.dufte)

    n_datasets = len(list(children(cwd_path("staged-decomp-values"))))
    n_models = max([len(list(children(dataset_path))) for _, dataset_path in children(cwd_path("staged-decomp-values"))])

    # TODO spacing between rows
    rowheight = 3
    fig, axs = plt.subplots(n_datasets, n_models + 1, sharey='row', figsize=(10, rowheight*n_datasets))

    for dataset_idx, (dataset_id, dataset_path) in enumerate(children(cwd_path("staged-decomp-values"))):

        # axs[dataset_idx, 1].set_ylabel("train error")

        # summary_ax = axs[0] if n_datasets == 1 else axs[dataset_idx, 0]
        summary_ax = axs[dataset_idx, 0]
        plot_summary(dataset_id, summary_ax)

        for model_idx, (model_id, _) in enumerate(children(dataset_path)):

            if n_datasets == 1:
                axs[model_idx].set_title(f"{model_id}")
            else:
                axs[0, model_idx+1].set_title(f"{model_id}")

            model_idx = model_idx + 1
            # ax = axs.flat[model_idx + dataset_idx * n_models]
            if n_datasets == 1:
                ax = axs[model_idx]
            else:
                ax = axs[dataset_idx, model_idx]

            for getter_id, label in zip(getters(), getters_labels()):
                if getter_id == "get_ensemble_bias":
                    continue
                plot_decomp_values(dataset_id, model_id, getter_id, ax, label=label)

            # ax.title(f"{decomp_id} on {dataset_id}")
            # ax.xlabel("Number of trees")
            # ax.ylabel("train error")
            # ax.set_title("title")

            # for ax in axs.flat:
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            # for ax in axs.flat:
            # ax.label_outer()
            # ax.set_xlabel("Number of trees")
            # ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

            # matplotx.line_labels(ax)  # line labels to the right
            matplotx.line_labels()  # line labels to the right
            # TODO shared legend

    fig.tight_layout()
    fig.savefig(cwd_path("plots", "bvd-decomps", "all.png"))

    # TODO this replaces staged errors
    # exp_ens_losses = [np.mean(decomp.get_expected_ensemble_loss(i)) for i in range(2,M)]
    # plt.plot(exp_ens_losses, label=decomp_id)

    # plt.legend()


if __name__ == "__main__":
    main()
