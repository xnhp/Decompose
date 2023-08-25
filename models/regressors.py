import logging

import numpy as np
import scipy
from scipy.optimize._linesearch import line_search
from sklearn.tree import DecisionTreeRegressor

from models.BaseHorizontalEnsemble import BaseHorizontalEnsemble
from models.util import deep_tree_params


# noinspection PyMethodMayBeStatic
class StandardRFRegressor(BaseHorizontalEnsemble):
    """ Vanilla Random Forest to be used for control/comparison """

    def __init__(self, base_estimator=DecisionTreeRegressor(
        # TODO squared error sure sounds like it would make sense here but have to take care when comparing to other things
        # TODO try different criteria?
        criterion="squared_error",  # !!
        **deep_tree_params
    ), bootstrap_rate=1.0, bootstrap_with_replacement=True):
        super().__init__(base_estimator, bootstrap_rate=bootstrap_rate,
                         bootstrap_with_replacement=bootstrap_with_replacement)

    def _combiner(self, preds):
        return np.mean(preds, axis=0)


class SubsamplingRFRegressor(StandardRFRegressor):

    def __init__(self, bootstrap_rate=0.5, bootstrap_with_replacement=False):
        super().__init__(bootstrap_rate=bootstrap_rate, bootstrap_with_replacement=bootstrap_with_replacement)


class SquaredErrorGradientRFRegressor(SubsamplingRFRegressor):

    def __init__(self, bootstrap_rate=None, bootstrap_with_replacement=False):
        super().__init__(bootstrap_rate=bootstrap_rate, bootstrap_with_replacement=bootstrap_with_replacement)
    #     if self.bootstrap_with_replacement is True:
    #         logging.warning("Using bootstrapping with replacement, this probably does not make sense with pseudotargets!")

    # want to know whether this achieves the same level of diversity faster than StandardRandomForest
    # the current BVD plot creates a new ensemble for each parameter value

    @staticmethod
    def _divergence_gradient(a, b):
        """ Gradient of (a-b)^2 wrt b """
        grad = -2 * (a - b)  # TODO negate?
        return grad

    @staticmethod
    def _divergence(a,b):
        # TODO use from class instead
        return (a-b)**2

    def estimator_targets(self, xs, ys):
        # output of ensemble built so far
        qbar = self.predict(xs)  # qbar_1^{M}
        # gradient of ensemble built so far
        g = self._divergence_gradient(ys, qbar)
        M = len(self.estimators_)
        # inverse of additive contribution of q_{M+1} to ensemble output qbar_1^{M+1}
        # return (M+1) * g + qbar  # => F_{M+1} \approx g
        return g + qbar  # => F_{M+1} \approx g
        # works better if we remove factor (M+1) I guess -- see line search
        # constant downscaling in front of g kind of works but does not fix bootstrapping issue,
        #   also works without
        # TODO seems to work better with factor (M+1) removed
        # sort of makes sense because it means each next tree receives even more weight
        # (why did we derive that anyway?)
        # -- could also try a sort of "line search" thing in direction of the gradient
        # TODO constantly high error with subsampling rate 0.5 (no replacement)

class SubsamplingSquaredErrorGradientRFRegressor(SquaredErrorGradientRFRegressor):

    def __init__(self, bootstrap_rate=0.6, bootstrap_with_replacement=False):
        super().__init__(bootstrap_rate, bootstrap_with_replacement)


class SimplifiedLineSearchSquaredErrorGradientRFRegressor(SquaredErrorGradientRFRegressor):
    def estimator_targets(self, xs, ys):
        qbar = self.predict(xs)  # qbar_1^{M}
        # gradient of ensemble built so far
        g = self._divergence_gradient(ys, qbar)
        M = len(self.estimators_)

        # perform line search to find best step size
        # this is lightly different from original gradient boosting formulation
        # but if we were to assume that q_{M+1} perfectly fitted the pseudo-targets,
        # this would work out mathematically
        def objective(beta):
            losses = self._divergence(ys, qbar + beta * g)
            return losses.sum()  # actually average but does not matter for minimisation

        def obj_grad(beta):
            # derivative of `objective` wrt beta
            v = -2 * g * ((-1)*beta*g - qbar + ys)
            return v.sum()

        beta_start = 0
        # direction = (M+1)
        direction = 1
        ls_result = line_search(objective, obj_grad, beta_start, direction)
        alpha = ls_result[0]
        beta = beta_start + alpha * direction
        return beta * g + qbar


class StupidLineSearchSquaredErrorGradientRFRegressor(SquaredErrorGradientRFRegressor):
    def estimator_targets(self, xs, ys):
        qbar = self.predict(xs)  # qbar_1^{M}
        # gradient of ensemble built so far
        g = self._divergence_gradient(ys, qbar)
        M = len(self.estimators_)

        # perform line search to find best step size
        # this is lightly different from original gradient boosting formulation
        # but if we were to assume that q_{M+1} perfectly fitted the pseudo-targets,
        # this would work out mathematically
        def objective(beta):
            losses = self._divergence(ys, qbar + beta * g)
            return losses.sum()  # actually average but does not matter for minimisation

        beta = stupid_line_search(objective, 0, 15, 100)
        return beta * g + qbar
def stupid_line_search(objective, min, max, n_points):
    linspace = np.linspace(min, max, n_points)
    min_idx = np.argmin([objective(value) for value in linspace])
    # TODO somehow always picks 0
    # also note that we're still not applying shrinking, which is probably more or less necessary
    # for boosting schemes?
    # TODO still does not work with bootstrapping
    # indicated here that subsampling without shrinkage "usually does poorly" in boosting
    # so omitting the (M+1) above effectively introduces shrinkage?
    # but irrc we've seen that subsampling / bootstrapping hurts it too
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regularization.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regularization-py
    return linspace[min_idx]


# # result = line_search(objective, gradient, point, direction)
# class LineSearchSquaredErrorGradientRFRegressor(SquaredErrorGradientRFRegressor):
#     def estimator_targets(self, xs, ys):
#         qbar = self.predict(xs)  # qbar_1^{M}
#         # gradient of ensemble built so far
#         g = self._divergence_gradient(ys, qbar)
#         M = len(self.estimators_)
#         # inverse of additive contribution of q_{M+1} to ensemble output qbar_1^{M+1}
#         # return (M + 1) * g + qbar  # => F_{M+1} \approx g
#
#         def objective(beta):
#             losses = self._divergence(ys, qbar + beta * (1/M) * (m - qbar))
#             return losses.sum()
#
#         def obj_grad(beta):
#             return -2 * g * (- g * beta - qbar + ys)
#
#         beta_start = 0  # seems reasonable, no change -- should warn though if this appears
#         direction = (M+1)  # TODO  (sth like max value?)
#         from scipy.optimize._linesearch import line_search
#         ls_result = line_search(objective, obj_grad, beta_start, direction)
#         alpha = ls_result[0]
#         beta = beta_start + alpha * direction
#
#         return beta * g + qbar
