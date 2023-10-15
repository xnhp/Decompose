import numpy as np


def _add_model_smoothing(pred, epsilon=1e-9):
    """
    Takes a 2-d numpy array of size N x K (number of examples by number of classes) and adds label noise. Takes
    an array with class probabilities for each example, increases the probability of each class by a small amount
    (epsilon), then re-normalises to ensure a valid probability distribution.

    Parameters
    ----------
    pred : numpy ndarray of shape (num_samples, num_labels)
        Predictions to which noise is to be added
    epsilon : float, optional (default=1e-9)
        size of the (unnormalized) increase in probability for each class

    Returns

    -------

    pred : numpy ndarray of shape (num_samples x num_labels)
        Predictions with added noise
    """
    if epsilon in [0., None]:
        return pred
    else:
        if epsilon < 0 or epsilon > 1.:
            raise ValueError("Value between 0 and 1 expected for smoothing factor")
        return (1 - epsilon) * pred + epsilon * (np.ones_like(pred)) * (1. / pred.shape[1])
