import math

import numpy as np
from scipy.stats import stats
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array

from decompose.bagging_ensembles import GeometricBaggingClassifier
from decompose.models.standard_rf_classifier import StandardRFClassifier
from skorch import NeuralNetClassifier
from torch import nn

class MyError(nn.Module):
    def __init__(self):
        super(MyError, self).__init__()

    def forward(self, inputs, targets):
        return nn.CrossEntropyLoss().forward(inputs, targets)

def make_geometric_nn_ensemble(base):
    # assumes cross-entropy loss in NN
    return GeometricBaggingClassifier(
        base_estimator=base,
        warm_start=True,
        smoothing_factor=1e-9
        # TODO n_estimators is still fixed
    )

class MLP(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.net = None

    def fit(self, xs, ys):

        input_dim = xs.shape[1]
        hidden_dim = 20
        output_dim = len(unique_labels(ys))

        xs = check_array(xs)

        module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        module = module.double()

        self.net = NeuralNetClassifier(
            module=module,
            max_epochs=100,
            lr=0.1,
            criterion=nn.CrossEntropyLoss(),
            verbose=False
        )

        return self.net.fit(xs, ys)

    def predict(self, xs):
        return self.net.predict(xs)

    def predict_proba(self, xs):
        return self.net.predict_proba(xs)


class NCLMLP(MLP):
    def __init__(self):
        super(NCLMLP, self).__init__()

    # for NCL, need prediction of ensemble
    # so have, to train all epochs of ensemble in a coordinated manner
    # -> will have write manual training loop after all
    # but probably not so bad

def lerp_b(m):

    b_min = 0
    b_max = 15
    m_max = 10
    m_eff = min(m, m_max)

    return (m_eff / m_max) * (b_max) * (-1)

    # min_m = 0
    # start = 0
    # stop = 15
    #
    # # import dvc.api
    # # params = dvc.api.params_show("params-sigmoid.yaml")
    # # max_m = params['max_m']
    # max_m = 15
    #
    # bs = - np.arange(start, stop, stop/(max_m+1))
    # assert len(bs) >= max_m + 1
    # if m < min_m:
    #     return bs[0]
    # if m > max_m:
    #     return bs[max_m]
    # return bs[m]

def transform_weights(weights, b):
    sigmoid_fn = make_sigmoid(b)
    # assert np.min(weights) >= 0
    # assert np.max(weights) <= 1
    return sigmoid_fn(weights)

def make_sigmoid(b):
    return lambda x: centered_sigmoid(b, x)

def centered_sigmoid(b, x):
    """
    sigmoid function with center at (0, 1/2), min 0, max 1
    -------

    """
    return sigmoid(a=0, b=b, k=1, x=x)

def sigmoid(a, b, k, x):
    return k/(1 + np.exp(a+b*x))


def normalize_weights(weights):
    return weights / weights.sum()


def y_drf_xu_chen(x):
    if (x < 1/2):
        return x**2
    else:
        return math.sqrt(x)

def ratio_incorrect_trees(tree_preds, truth):
    incorrect = tree_preds != truth  # binary vector with 1 where incorrect
    means = np.mean(incorrect, axis=0)  # sum over trees # TODO verify this is the right axis
    return means

def ratio_votes_next_best(tree_preds, truth):
    cat = np.concatenate(([truth], tree_preds), axis=0)
    r = np.apply_along_axis(ratio_votes_next_best_el, 0, cat)
    return r


def sort(arr, axis):
    sorted_indices = np.argsort(arr, axis=axis)
    sorted_arr = arr[np.arange(arr.shape[0])[:, None], sorted_indices]
    return sorted_arr

def ratio_votes_next_best_el(single_tree_preds):
    # first el is always ground truth
    if len(single_tree_preds) == 1:  # no actual tree preds
        return 0
    gt = single_tree_preds[0]
    tree_preds = single_tree_preds[1:]
    # mask out gt
    tree_preds_for_other_classes = tree_preds[tree_preds != gt]
    if len(tree_preds_for_other_classes) == 0:
        return 0  # all voted for gt
    # otherwise, get most common class
    mode = stats.mode(tree_preds_for_other_classes).mode[0]
    votes_for_mode = sum(tree_preds_for_other_classes == mode)
    return votes_for_mode / len(single_tree_preds - 1)

    # u, c = np.unique(tree_preds, return_counts=True)
    # sinds = np.argsort(-c)
    # if sinds.shape[0] < 2:
    #     return 0

    pass
    return 0

    # if len(u) <= 1:
    #     return 0  # only one value


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


