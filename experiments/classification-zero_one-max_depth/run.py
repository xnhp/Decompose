from sklearn.ensemble import RandomForestClassifier

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from myexperiments import MyExperiment
from utils import CustomMetric, DatasetId

custom_metrics = {
    CustomMetric.MEMBER_DEVIATION,
    CustomMetric.EXP_MEMBER_LOSS
}


dataset = load_standard_dataset(DatasetId.MNIST, frac_training=0.5, label_noise=0.3)

exp = MyExperiment(
    __file__,
    dataset,
    BVDExperiment(
        RandomForestClassifier(n_estimators=5),
        loss="zero_one",
        parameter_name="max_depth",
        parameter_values=range(1, 68, 4),
        save_decompositions=True
    ),
    n_trials=3,
    custom_metrics=custom_metrics
)

exp.plot(custom_metrics=custom_metrics)
