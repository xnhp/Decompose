import logging

import numpy as np
import scipy
from sklearn.tree import DecisionTreeClassifier

from decompose.BaseHorizontalEnsemble import BaseHorizontalEnsemble
from decompose.util import deep_tree_params


def _drf_sample_weights(tree_preds, truth):
    incorrect = tree_preds != truth  # binary vector with 1 where incorrect
    means = np.mean(incorrect, axis=0)  # sum over trees # TODO verify this is the right axis
    return means / means.sum()


class StandardRFClassifier(BaseHorizontalEnsemble):
    def __init__(self, base_estimator=DecisionTreeClassifier(
        criterion="gini",
        **deep_tree_params
    )):
        super().__init__(base_estimator, bootstrap_rate=1.0)

    def _combiner(self, preds):
        modes = scipy.stats.mode(preds, axis=0)
        return modes.mode[0]

class DRFWeightedBootstrapRFClassifier(StandardRFClassifier):
    """ Variant of Vanilla RF that uses bootstrap weighting according to Bernard2012
        Differences to Bernard2012: They additionally use...
        - weighted splitting criterion
        - another method for selecting the number of features to test
        - determine the weights on out-of-bag trees
        - use the weights *both* for weighted bootstrap sampling and weighted split selection
     """

    def _bootstrap_sample_weights(self, data, truth):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        # all the predictions we have up to here
        tree_preds = self._tree_preds()
        return _drf_sample_weights(tree_preds, truth)


class DRFWeightedFitOOBRFClassifier(StandardRFClassifier):

    def __init__(self, base_estimator=DecisionTreeClassifier(
        criterion="gini",
        **deep_tree_params
    )):
        super().__init__(base_estimator)
        self.drf_sample_weights = []

    def get_oob_trees(self):
        current_bootstrap_indices = self.bootstrap_indices[-1]  # assume that this is executed *after* bootstrapping
            # has been done for next tree
        r = []
        for tree_index, tree_oob_indices in enumerate(self.oob_indices): # oob_indices of previous trees
            weights_to_modify_indices = np.intersect1d(tree_oob_indices, current_bootstrap_indices)
            r.append(weights_to_modify_indices)  # may be empty if tree is not an oob tree -- but simpler to just carry along the empty case
        return r

    def fit_sample_weights(self, bootstrap):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        xs, ys = bootstrap

        # only modify weight if at least one oob tree is available
        # i.e. if example index is in oob_indices of previous tree
        # i.e. get trees t' for which weights_to_modify_indices is not empty
        # ! this is different for each example!
        # -> i.e. need a different predict call for each example!
        oob_trees = self.get_oob_trees()  # this at least computes the set intersections only once
        # just have all trees predict on all points, then pick later,
        # should be worth it assuming have many oob samples
        tree_preds = np.array([est.predict(xs) for est in self.estimators_])
        def oob_drf_weight(x_idx, y):
            oob_tree_indices = np.where(x_idx == oob_trees)[0]
            # oob_tree_indices = list(filter(lambda weights_to_modify: x_idx in weights_to_modify, oob_trees))
            if len(oob_tree_indices) > 0:
                oob_tree_preds = tree_preds[oob_tree_indices]
                return _drf_sample_weights(oob_tree_preds, [y])
            else:
                # old weight
                # but what is the very first weight?
                if (len(self.estimators_) -1 ) == 0:
                    return [1/2]
                return self.drf_sample_weights[-1][x_idx]

        new_weights = []
        for x_idx in np.arange(xs.shape[0]):
            y = ys[x_idx]
            new_weights.append(oob_drf_weight(x_idx, y))
        new_weights = np.array(new_weights).squeeze()

        # v_oob_drf_weight = np.vectorize(oob_drf_weight)
        # new_weights = v_oob_drf_weight(np.arange(xs.shape[0]), ys)
        self.drf_sample_weights.append(new_weights)
        assert new_weights.shape[0] == xs.shape[0]
        return new_weights

        # idea: each tree predicts only its `weights_to_modify` indices

        # weights_to_modify_indices := oob_indices(t') \cap bootstrap_indices(t)

        # compute ensemble prediction of these trees
        # oob_tree_preds = _tree_preds(xs[weights_to_modify_indices], tree_indices)
        # new_oob_weights = _drf_sample_weights(oob_tree_preds, ys)

        # ↝ apply
        # new_sample_weights[weights_to_modify_indices] = new_oob_weights  # shapes should work out

        # -- else use old weight!
        # use_old_weights_indices := np.arange(xs.shape[0]) - weights_to_modify_indices  # (np.setdiff1d)
        # old_sample_weights = drf_sample_weights(t-1)
        # ↝ apply
        # new_sample_weights[use_old_weights_indices] = old_sample_weights[use_old_weight_indices]


class DRFWeightedFitRFClassifier(StandardRFClassifier):
    """Variant of vanilla RF that uses fit example weighted according to Bernard2012"""
    def fit_sample_weights(self, bootstrap):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        xs, ys = bootstrap
        tree_preds = self._tree_preds()
        return _drf_sample_weights(tree_preds, ys)


class AdjustedDRFWeightedRFClassifier(StandardRFClassifier):
    def _bootstrap_sample_weights(self, data, truth):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        # 1/M*(#incorrect-trees)
        tree_preds = self._tree_preds()
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

    def _bootstrap_sample_weights(self, data, truth):
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

    def _bootstrap_sample_weights(self, data, truth):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        tree_preds = self._tree_preds()
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


