import copy
import logging
from functools import cache

import numpy as np
import numpy.random

from models.adaboost import ModelWrapper
from models.caching import HashingWrapper


# noinspection PyAttributeOutsideInit,DuplicatedCode,PyMethodMayBeStatic
class BaseHorizontalEnsemble(object):
    """
    Customization of Adaboost (adaboost.py).
    Base methods might still implement Adaboost behaviour
    """

    def __init__(self
                 , base_estimator=None
                 , n_estimators=5  # potentially overwritten by BVDExperiment
                 , bootstrap_rate=None  # no bootstrapping whatsoever if None
                 , bootstrap_with_replacement=True
                 , warm_start=False  # handled in BVDExperiment
                 ):
        # TODO pass in / use bootstrap parameters
        # todo some of these parameters are only relevant to adaboost
        if base_estimator is None:
            raise ValueError("Base estimator must be supplied")
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.warm_start = warm_start
        self.bootstrap_rate = bootstrap_rate
        self.bootstrap_with_replacement = bootstrap_with_replacement
        self.pred_cache = {}

    def fit(self, xs, ys):
        """
        Fits the ensemble by training base estimators sequentially.

        Parameters
        ----------
        data - ndarray of shape (n_examples, n_features)
            Training data features
        labels - ndarray of shape (n_examples)
            Training data labels

        Returns
        -------
        None

        """
        ys = self._init_fit(xs, ys)

        if self.bootstrap_rate is None:
            logging.warning("No bootstrapping (not requested)")

        if self.bootstrap_with_replacement is False:
            logging.warning("bootstrapping without replacement")

        # construct ensemble horizontally
        # TODO factor out when doing vertical groptimisation
        # while len(self.estimators_) < self.n_estimators:
        for i in range(self.n_estimators):
            logging.debug("fitting estimator {i} of {self.n_estimators}")

            estimator = ModelWrapper(copy.deepcopy(self.base_estimator))

            if self.bootstrap_rate is None:
                bootstrap = xs, ys  # no bootstrapping, use whole dataset
            else:
                bootstrap = self.bootstrap_(xs, ys)
            effective_xs, effective_ys = bootstrap

            if len(self.estimators_) == 0:
                # no pseudo-targets for very first estimator
                # -- leave unchanged
                pass
            else:
                # use subsample to fit base learner and compute model update
                effective_ys = self.estimator_targets(effective_xs, effective_ys)  # "pseudo-targets"

            self.sample_weight = np.ones_like(effective_ys)
            # if len(self.estimators_) == 0:
            #     # always uniform sample weights for very first estimator
            # else:
            #     self.sample_weight = self.fit_sample_weights(bootstrap)

            estimator.model.fit(effective_xs, effective_ys)

            estimator_weight = self.estimator_weight_(effective_xs, effective_ys, estimator)
            estimator.weight = estimator_weight
            self.estimator_weights.append(estimator_weight)

            self.estimators_.append(estimator)

    def bootstrap_(self, data, truth):
        """ Return tuple of bootstrap samples for data, truth. Pure. """
        bootstrap_idx = self._bootstrap_indices(data, truth)
        return data[bootstrap_idx], truth[bootstrap_idx]

    def _bootstrap_indices(self, data, truth):
        """ Determine (indices of) bootstrap sample to be used for next estimator. Pure. """
        # rng = np.random.default_rng()
        # return rng.integers(low=0, high=n, size=int(n * self.bootstrap_rate))  # sample n indices with replacement
        n = data.shape[0]
        bootstrap_probs = self.bootstrap_sample_weights(data, truth)
        return numpy.random.choice(n,
                                   size=int(n * self.bootstrap_rate),
                                   replace=self.bootstrap_with_replacement,
                                   p=bootstrap_probs)

    def bootstrap_sample_weights(self, data, truth):
        """Sample weights used for picking bootstrap samples. Pure."""
        logging.debug("using uniform bootstrap sample weights")
        return None  # consistent if supplied to numpy.random.choice

    def fit_sample_weights(self, bootstrap):
        """
        Sample weights used for fitting the estimator. Pure.
        TODO needs to be revisited anyway because weighting and bootstrapping (with replacement) probably does
            not make much sense from a boosting perspective (i.e. try without replacement, which is akin to "boosting
            by resampling"
            so maybe just move this method to act on full sample instead
        """
        # Default implementation
        logging.debug("using uniform fit sample weights")
        boot_xs, boot_ys = bootstrap
        return np.ones_like(boot_ys)

    def estimator_targets(self, xs, ys):
        """ Determine pseudo-targets. Pure. """
        # Default implementation
        return ys

    def estimator_weight_(self, xs, ys, new_estimator):
        """
        determine weight (shrinking). Pure.
        """
        # Default implementation
        logging.info("using uniform estimator weight")
        return 1

    def _init_fit(self, data, labels):
        """
        Just boilerplate / management stuff
        """
        # labels = _validate_labels_plus_minus_one(labels)
        # start with uniform weights over training data unless explicitly given
        if not hasattr(self, "sample_weight"):
            self.sample_weight = np.ones(data.shape[0]) / data.shape[0]
        if self.warm_start:
            if not hasattr(self, "data"):
                self.data = data
                self.labels = labels
            else:
                if (self.data != data).all() or (self.labels != labels).all():
                    ValueError("Must receive same data on each iteration")
        if not hasattr(self, "estimators_"):
            self.estimators_ = []
            self.estimator_weights = []
        return labels

    def _tree_preds(self, xs, M=None):
        hxs = HashingWrapper(xs)
        try:
            cached_for_xs = self.pred_cache[hxs]
        except KeyError:
            self.pred_cache[hxs] = []

        if M is None:
            M = len(self.estimators_)

        # extend cached sequence if needed
        for i in range(len(self.pred_cache[hxs]), M):
            estimator = self.estimators_[i]
            preds = estimator.predict(xs)
            self.pred_cache[hxs].append(preds)

        # return np.array([self._estimator_pred(estimator, xs) for estimator in self.estimators_[:M]])
        return np.array(self.pred_cache[hxs][:M])

    def _combiner(self, preds):
        # TODO cache combiner?
        raise NotImplementedError

    def staged_predict(self, data):
        for M in range(0, len(self.estimators_)):
            yield self._combiner(self._tree_preds(data, M))

    def predict(self, data):
        # just need to apply combiner across individual estimators
        # in case of squared loss, combiner is just arithmetic mean
        return self._combiner(self._tree_preds(data))
