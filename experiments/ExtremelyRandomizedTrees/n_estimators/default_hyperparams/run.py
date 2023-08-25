import os

from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from myexperiments import MyExperiment
from utils import DatasetId, CustomMetric

cmetrics = {
    CustomMetric.MEMBER_DEVIATION,
    CustomMetric.EXP_MEMBER_LOSS,
    CustomMetric.ENSEMBLE_VARIANCE,
    CustomMetric.ENSEMBLE_BIAS
}

dataset = load_standard_dataset(DatasetId.MNIST, frac_training=0.5, label_noise=0.3)


# compare to classification-zero_one

exp = MyExperiment(
    __file__,
    dataset,
    BVDExperiment(
        ExtraTreesClassifier(),
        loss="zero_one",
        parameter_name="n_estimators",
        parameter_values=range(1, 20, 2),
    ),
    n_trials=3,
    custom_metrics=cmetrics,
)

exp.plot(custom_metrics=cmetrics)
# TODO seems buggy