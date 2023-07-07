from sklearn.ensemble import RandomForestRegressor

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from myexperiments import MyExperiment
from utils import CustomMetric

cmetrics = {
    CustomMetric.ENSEMBLE_VARIANCE,
    CustomMetric.ENSEMBLE_BIAS
}
exp = MyExperiment(
    __file__,
    BVDExperiment(
        RandomForestRegressor(),
        loss="squared",
        parameter_name="max_depth",
        parameter_values=range(1, 68, 4),
    ),
    n_trials=3,
    custom_metrics=cmetrics
)

exp.plot(custom_metrics=cmetrics)
