from sklearn.ensemble import RandomForestRegressor

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from myexperiments import MyExperiment
from utils import CustomMetric

identifier = "covariance"
exp = MyExperiment(
    identifier,
    load_standard_dataset("california", frac_training=0.5, normalize_data=True),
    BVDExperiment(
        RandomForestRegressor(),
        loss="squared",
        parameter_name="n_estimators",
        parameter_values=[15],
        save_decompositions=True,
        decompositions_prefix=identifier,
    ),
    n_trials=3,
    custom_metrics={
        CustomMetric.COVMAT
    }
)

exp.plot_mat_heat(CustomMetric.COVMAT)
