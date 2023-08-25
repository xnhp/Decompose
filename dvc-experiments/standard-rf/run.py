import os
import warnings

from ruamel.yaml import YAML

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from models.regressors import StandardRFRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    yaml = YAML(typ="safe")
    params = yaml.load(open("params.yaml", encoding="utf-8"))  # use the params show thing instead
    model = StandardRFRegressor()
    bvd_exp = BVDExperiment(model, **params['bvd_config'])
    dataset = load_standard_dataset(params['data']['dataset_name'], frac_training=params['data']['frac_training'])
    results = bvd_exp.run_experiment(*dataset, n_trials=params['run_experiment_config']['n_trials'], custom_metrics={})
    results.save_results(os.path.join(os.path.dirname(__file__), "results.pkl"))

if __name__ == "__main__":
    main()

#
# plot_params = {
#     "start_at": 5
# }
#
# plot_staged_errors(exps, lambda exp: exp.get_results.staged_errors, title="Test", **plot_params)
