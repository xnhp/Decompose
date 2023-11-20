import scipy
from sklearn.tree import DecisionTreeClassifier

from decompose.BaseHorizontalEnsemble import BaseHorizontalEnsemble
from decompose.util import deep_tree_params


class StandardRF(BaseHorizontalEnsemble):
    def __init__(self, base_estimator=DecisionTreeClassifier(
        criterion="gini",
        **deep_tree_params
    )):
        super().__init__(base_estimator, bootstrap_rate=1.0)

    def _combiner(self, preds):
        modes = scipy.stats.mode(preds, axis=0)
        return modes.mode[0]
