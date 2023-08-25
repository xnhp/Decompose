import os
import warnings

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from models.regressors import StandardRFRegressor, SquaredErrorGradientRFRegressor, \
    SimplifiedLineSearchSquaredErrorGradientRFRegressor, StupidLineSearchSquaredErrorGradientRFRegressor
from myexperiments import MyExperiment, plot_staged_errors

warnings.simplefilter(action='ignore', category=FutureWarning)

shared_bvd_config = {
    "loss": "squared",
    "parameter_name": "n_estimators",
    "parameter_values": [150],
    "save_decompositions": True,
    "trials_progress_bar": True
}

shared_myexp_config = {
    "n_trials": 1,  # trials for bvd decomp evaluation
    "custom_metrics": {
        # CustomMetric.MEMBER_DEVIATION,
        # CustomMetric.EXP_MEMBER_LOSS,
        # CustomMetric.ENSEMBLE_VARIANCE,
        # CustomMetric.ENSEMBLE_BIAS
    },
    # "debug":True
}

dataset_name = "california"
dataset = load_standard_dataset(dataset_name, frac_training=3/4)

my_exp = lambda s, c: MyExperiment(
    os.path.join(os.path.dirname(__file__), os.path.join(s, "foo")),
    dataset,
    BVDExperiment(c, **shared_bvd_config),
    **shared_myexp_config
)

exps = [
    my_exp("control", StandardRFRegressor()),
    my_exp("boosted", SquaredErrorGradientRFRegressor()),
    my_exp("simple-ls", SimplifiedLineSearchSquaredErrorGradientRFRegressor()),
    my_exp("stupid-ls", StupidLineSearchSquaredErrorGradientRFRegressor())
]


plot_params = {
    "start_at": 5
}
# TODO port staged_errors to other decomposition object
plot_staged_errors(exps, lambda exp: exp.get_results.staged_errors, title="Test", **plot_params)
