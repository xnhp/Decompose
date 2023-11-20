import numpy as np
from sklearn.tree import DecisionTreeClassifier

from decompose.classifiers import ratio_incorrect_trees, normalize_weights
from decompose.models.standard_rf_classifier import StandardRFClassifier
from decompose.util import deep_tree_params


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
                weights = ratio_incorrect_trees(oob_tree_preds, [y])
                return normalize_weights(weights)
            else:
                # old weight
                # but what is the very first weight?
                if (len(self.estimators_) -1 ) == 0:
                    return normalize_weights(np.array([1/2]))
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
