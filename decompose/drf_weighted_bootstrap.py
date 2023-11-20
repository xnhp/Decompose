from decompose.classifiers import normalize_weights, ratio_incorrect_trees
from decompose.models.standard_rf_classifier import StandardRFClassifier


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
        return normalize_weights(ratio_incorrect_trees(tree_preds, truth))
