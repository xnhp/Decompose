import os
import warnings

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from decompose.utils import get_model

warnings.simplefilter(action='ignore', category=FutureWarning)

import dvc.api


def main():
    params = dvc.api.params_show(stages="train")
    model = get_model(params['model'])
    bvd_exp = BVDExperiment(model, **params['bvd_config'])
    dataset = load_standard_dataset(params['data']['dataset_name'], frac_training=params['data']['frac_training'])
    results = bvd_exp.run_experiment(*dataset, n_trials=params['run_experiment_config']['n_trials'], custom_metrics={})
    results.save_results(os.path.join(os.path.dirname(__file__), params['model'] + "/results.pkl"))


if __name__ == "__main__":
    main()

#
# plot_params = {
#     "start_at": 5
# }
#
# plot_staged_errors(exps, lambda exp: exp.get_results.staged_errors, title="Test", **plot_params)
