from decompose.classifiers import ratio_incorrect_trees, normalize_weights
from decompose.models.standard_rf import StandardRF


class DRFWeightedFitRFClassifier(StandardRF):
    """Variant of vanilla RF that uses fit example weighted according to Bernard2012"""
    def fit_sample_weights(self, bootstrap):
        if len(self.estimators_) == 0:
            # first estimator
            return None
        xs, ys = bootstrap
        tree_preds = self._tree_preds()
        weights = ratio_incorrect_trees(tree_preds, ys)
        return normalize_weights(weights)
