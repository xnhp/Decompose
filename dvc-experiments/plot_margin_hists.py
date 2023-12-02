import pickle

from matplotlib import pyplot as plt

from decompose.dvc_utils import cwd_path
from decompose.utils import data_model_foreach


def mg_distr_by_example(M, decomp):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    e = decomp.error_function((), (0), M)
    mg_per_ex = e.mean(axis=0)
    below_half = mg_per_ex[mg_per_ex < 0.5]
    above_half = mg_per_ex[mg_per_ex >= 0.5]
    ax.hist(below_half, bins=100, alpha=0.5, label="below half", color="green")
    ax.hist(above_half, bins=100, alpha=0.5, label="above half", color="red")
    ax.axvline(1 / 2, color="black", label="0.5", alpha=0.5)
    ax.set_xlabel("Margin (ratio incorrect trees)")
    ax.set_title(f"margin distr. by example for M={M}")
    return fig

def plot_margin_distr_by_example(dataset_id, model_id):
    decomp_path = cwd_path("decomps", dataset_id, model_id + ".pkl")
    with open(decomp_path, "rb") as f:
        decomp = pickle.load(f)

    ms = [2, 5, 10, 20, 50, 100, 150]
    for m in ms:
        fig = mg_distr_by_example(m, decomp)
        fig.savefig(cwd_path("plots", "margin_hists", dataset_id, model_id, f"by-example-{m}.png"))


if __name__ == "__main__":
    # data_model_foreach(cwd_path("decomps"), plot_margin_distr_by_example)

    binary_datasets = [
        'qsar-biodeg', "diabetes", "bioresponse", "spambase-openml"
    ]
    nonbinary_datasets = ['digits', 'mnist_subset', 'cover']

    all_datasets = binary_datasets + nonbinary_datasets

    tasks = [
        {
            'out_path': "plots/margin_hists/drf_sigmoid/",
            'datasets': ["bioresponse"],
            "models": [
                'standard_rf',
                'drf_weighted_bootstrap',
                'capped_sigmoid',
                'capped_lerped_sigmoid'
            ]
        }
    ]

    for task in tasks:
        out = cwd_path(task['out_path'])
        for dataset_id in task['datasets']:
            for model_id in task['models']:
                plot_margin_distr_by_example(dataset_id, model_id)

