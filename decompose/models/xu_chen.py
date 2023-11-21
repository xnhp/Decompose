import numpy as np

from decompose.classifiers import ratio_incorrect_trees, y_drf_xu_chen, normalize_weights
from decompose.models.standard_rf import StandardRF


class XuChenWeightedBootstrapRFClassifier(StandardRF):
    def _bootstrap_sample_weights(self, data, truth):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        # all the predictions we have up to here
        tree_preds = self._tree_preds()
        w = ratio_incorrect_trees(tree_preds, truth)
        w = np.array([y_drf_xu_chen(x) for x in w])
        return normalize_weights(w)
