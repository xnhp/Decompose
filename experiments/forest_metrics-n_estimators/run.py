from sklearn.ensemble import RandomForestRegressor

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from myexperiments import MyExperiment
from utils import CustomMetric, DatasetId

cmetrics = {
    CustomMetric.ENSEMBLE_VARIANCE,
    CustomMetric.ENSEMBLE_BIAS
}
dataset = load_standard_dataset(DatasetId.MNIST, frac_training=0.5, label_noise=0.3)
exp = MyExperiment(
    __file__,
    dataset,
    BVDExperiment(
        RandomForestRegressor(),  # TODO this does not make sense with MNIST
        loss="squared",
        parameter_name="n_estimators",
        parameter_values=range(1, 20, 2),
        save_decompositions=True,
    ),
    n_trials=3,
    custom_metrics=cmetrics
)

exp.plot(custom_metrics=cmetrics)
