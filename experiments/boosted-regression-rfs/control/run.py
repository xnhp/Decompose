# adaboost_ens = AdaBoost(DecisionTreeClassifier(max_leaf_nodes=MAX_LEAF),
#                         n_estimators=N_ESTIMATORS,
#                         shrinkage=SHRINKAGE)
#
# classification_experiment = BVDExperiment(adaboost_ens, bootstrap=False,
#                                           n_samples=.9, ensemble_warm_start=False, loss="exponential",
#                                           save_decompositions=True,
#                                           decompositions_prefix=DECOMPOSITIONS_FILE_PREFIX + "_" + dataset,
#                                           trials_progress_bar=True)
#
# logging.debug(
#     f"Starting experiment on {adaboost_ens}\nMax leaf nodes: {MAX_LEAF}\nn_estimators: {N_ESTIMATORS}\nshrinkage:{SHRINKAGE}\nfrac_train:{FRAC_TRAIN}")
# classification_experiment.run_experiment(train_X,
#                                          train_y,
#                                          test_X,
#                                          test_y,
#                                          n_trials=N_TRIALS)
from sklearn.ensemble import RandomForestRegressor

from decompose import BVDExperiment
from decompose.data_utils import load_standard_dataset
from models.regressors import StandardRFRegressor, SquaredErrorGradientRFRegressor
from myexperiments import MyExperiment
from utils import CustomMetric

exp = MyExperiment(
    __file__,
    load_standard_dataset("california", frac_training=0.5, normalize_data=True),
    BVDExperiment(
        StandardRFRegressor(),
        loss="squared",
        parameter_name="n_estimators",
        parameter_values=range(1,17,1),
        save_decompositions=False,
        trials_progress_bar=False
    ),
    n_trials=3,
    custom_metrics={
        # CustomMetric.MEMBER_DEVIATION,
        # CustomMetric.EXP_MEMBER_LOSS,
        # CustomMetric.ENSEMBLE_VARIANCE,
        # CustomMetric.ENSEMBLE_BIAS
    },
    debug=True
)

exp.plot()
