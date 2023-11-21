import numpy as np

from decompose.classifiers import ratio_incorrect_trees, ratio_votes_next_best, transform_weights, lerp_b, \
    normalize_weights
from decompose.models.standard_rf import StandardRF


class DRFGoodWeightedBootstrapRFClassifier(StandardRF):
    """
        Constant above 1/2 -- only applies to binary classif
     """

    def _bootstrap_sample_weights(self, data, truth):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        # all the predictions we have up to here
        tree_preds = self._tree_preds()
        w = ratio_incorrect_trees(tree_preds, truth)

        # this is now per example
        thresh = ratio_votes_next_best(tree_preds, truth)

        w = transform_weights(w - thresh, lerp_b(len(self.estimators_)))
        w = np.clip(w, 0, 1/2) # always above 1/2 exactly beyond threshold
        return normalize_weights(w)
