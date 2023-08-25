import logging

import numpy as np
import scipy
from sklearn.tree import DecisionTreeClassifier

from models.BaseHorizontalEnsemble import BaseHorizontalEnsemble
from models.util import deep_tree_params


class StandardRFClassifier(BaseHorizontalEnsemble):
    def __init__(self, base_estimator=DecisionTreeClassifier(
        criterion="gini",
        **deep_tree_params
    )):
        super().__init__(base_estimator, bootstrap_rate=1.0)

    def _combiner(self, preds):
        modes = scipy.stats.mode(preds, axis=0)
        return modes.mode[0]

class DRFWeightedRFClassifier(StandardRFClassifier):
    """ Variant of Vanilla RF that uses bootstrap weighting according to Bernard2012
        Differences to Bernard2012: They additionally use...
        - weighted splitting criterion
        - another method for selecting the number of features to test
        - determine the weights on out-of-bag trees
     """

    def bootstrap_sample_weights(self, data, truth):
        # 1/M*(#incorrect-trees)
        tree_preds = self._tree_preds(data)
        if len(self.estimators_) == 0:
            # first estimator
            return None
        incorrect = tree_preds != truth  # binary vector with 1 where incorrect
        means = np.mean(incorrect, axis=0)  # sum over trees # TODO verify this is the right axis
        return means / means.sum()

class AdjustedDRFWeightedRFClassifier(StandardRFClassifier):
    def bootstrap_sample_weights(self, data, truth):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        # 1/M*(#incorrect-trees)
        tree_preds = self._tree_preds(data)
        ens_preds = self.predict(data)
        ens_incorrect = ens_preds != truth
        ens_correct = ens_preds == truth
        trees_incorrect = tree_preds != truth  # binary vector with 1 where incorrect

        ratio_trees_incorrect = np.mean(trees_incorrect, axis=0)  # mean over trees

        rate = 1/4 # assumed to be in [0,1]  # does not work as well with 1/2
        # left
        to_base = ratio_trees_incorrect -1
        scaled = to_base * rate  # here we downscale
        from_base = scaled + 1
        left = from_base * ens_incorrect  # only left side
        # right
        right = ratio_trees_incorrect * (1+rate) * ens_correct  # here we upscale
        res = left + right  # should be mututally exclusive
        return res / res.sum()

class SimpleWeightedRFClassifier(StandardRFClassifier):
    """ Use bootstrap weighting based on correctness of ensemble """

    def bootstrap_sample_weights(self, data, truth):
        if len(self.estimators_) == 0:
            # first estimator
            ones = np.ones_like(truth)
            return ones / ones.sum()
        ens_preds = self.predict(data)
        ens_incorrect = ens_preds != truth
        ens_correct = ens_preds == truth

        # perfect fit
        if ens_incorrect.sum() == 0:
            # logging.warning(f"perfect fit at {len(self.estimators_)} estimators, continuing with uniform weights")
            ones = np.ones_like(truth)
            return ones / ones.sum()  # uniform 1/n

        rate = 1/4

        weights = np.full_like(truth, 1/2, dtype=np.double)
        additive = rate * ens_incorrect
        subtractive = rate * ens_correct

        weights = weights + additive - subtractive

        return weights / weights.sum()  # weights will need to sum to one to satisfy implementation

        # correct points will have weight 0
        # incorrect points will have weight 1/(#incorrect-pts)
        # assert ens_incorrect.sum() != 0
        # incorrect_weights = ens_incorrect / ens_incorrect.sum()
        # try adjusting the weights only a little
        # return np.ones_like(truth) +

class DiversityEffectWeightedRFClassifier(StandardRFClassifier):

    def bootstrap_sample_weights(self, data, truth):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        tree_preds = self._tree_preds(data)
        ens_preds = self.predict(data)
        incorrect = (tree_preds != truth)*1.0  # L(y, q_i)  # binary vector with 1 where incorrect
        yqbar = (ens_preds != truth)*1.0
        effects = (incorrect - yqbar)
        means = np.mean(effects, axis=0)  # sum over trees # TODO verify this is the right axis
        # this is exactly diversity-effect / ambiguity-effect
        # TODO why does this work? and why is DRF still better?
        if means.sum() == 0:
            return None
        r = 1 + means
        return r / r.sum()

    # def bootstrap_sample_weights(self, data, truth):
    #     if len(self.estimators_) == 0:
    #         # first estimator
    #         ones = np.ones_like(truth)
    #         return ones / ones.sum()
    #     qbar = self.predict(data)
    #
    #     qis = self._tree_preds(data)  # all individual tree predictions
    #     qiy = qis != truth
    #     qiqbar = qis != qbar  # should be (y, qi), need to check shape
    #
    #     fac = truth * qbar # TODO but assuming these are -1, 1
    #
    #     return np.mean(qiy - fac * qiqbar)


