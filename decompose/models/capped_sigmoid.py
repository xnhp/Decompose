import numpy as np

from decompose.classifiers import ratio_incorrect_trees, transform_weights, normalize_weights
from decompose.models.standard_rf import StandardRF


class CappedSigmoid(StandardRF):
    def _bootstrap_sample_weights(self, data, truth):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        # all the predictions we have up to here
        tree_preds = self._tree_preds()
        w = ratio_incorrect_trees(tree_preds, truth)
        b = 15
        # b = len(self.estimators_)
        w = transform_weights(w - 1/2, b)
        w = np.clip(w, 0, 1/2) # always above 1/2 exactly beyond threshold
        return normalize_weights(w)
