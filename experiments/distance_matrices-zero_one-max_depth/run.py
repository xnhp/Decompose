from sklearn.ensemble import RandomForestClassifier

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from myexperiments import MyExperiment
from utils import CustomMetric, DatasetId

# Classification / Discrete

identifier = "distance_matrices-zero_one-max_depth"
exp = MyExperiment(
    identifier,
    load_standard_dataset(DatasetId.WINE, frac_training=0.5, label_noise=0.3),
    BVDExperiment(
        RandomForestClassifier(),
        loss="zero_one",
        parameter_name="max_depth",
        parameter_values=range(1, 32, 4),
        save_decompositions=True,
        decompositions_prefix=identifier + "-decomp",
    ),
    n_trials=3,
    custom_metrics={
        CustomMetric.DISTMAT_DHAT,
        CustomMetric.DISTMAT_DISAGREEMENT
    }
)

exp.plot_mat_mds(CustomMetric.DISTMAT_DHAT, "Dhat")
exp.plot_mat_mds(CustomMetric.DISTMAT_DISAGREEMENT, "disagreement")

exp.plot_mat_heat(CustomMetric.DISTMAT_DHAT)
exp.plot_mat_heat(CustomMetric.DISTMAT_DISAGREEMENT)
