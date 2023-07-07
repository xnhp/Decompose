from sklearn.ensemble import RandomForestClassifier

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from myexperiments import MyExperiment
from utils import CustomMetric, DatasetId

# Classification / Discrete

exp = MyExperiment(
    __file__,
    BVDExperiment(
        RandomForestClassifier(),
        loss="zero_one",
        parameter_name="max_depth",
        parameter_values=range(1, 32, 4),
        save_decompositions=True,
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
