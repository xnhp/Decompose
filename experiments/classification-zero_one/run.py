from sklearn.ensemble import RandomForestClassifier

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from myexperiments import MyExperiment
from utils import CustomMetric, DatasetId

custom_metrics = {
    CustomMetric.MEMBER_DEVIATION,
    CustomMetric.EXP_MEMBER_LOSS
}

exp_name = "classification-zero_one"

exp = MyExperiment(
    exp_name,
    load_standard_dataset(
        DatasetId.MNIST,
        frac_training=0.5,
        label_noise=0.3
    ),
    BVDExperiment(
        RandomForestClassifier(n_estimators=5),
        loss="zero_one",
        parameter_name="n_estimators",
        parameter_values=range(1, 20, 2),
        save_decompositions=True,
        decompositions_prefix=exp_name + "_decomp"
    ),
    n_trials=3,
    custom_metrics=custom_metrics
)

exp.plot(custom_metrics=custom_metrics)