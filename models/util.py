import numpy as np

# parameters for deep trees as used in vanilla random forests
# used as parameters for DecisionTreeClassifier, DecisionTreeRegressor
deep_tree_params = {
    # splitter="random",  # samples random splits, then chooses the best of these -- also possible but not vanilla RF
    # params below taken from sklearn RandomForestRegressor
    "max_features": "sqrt",  # `None` or `1.0` will consider all features.
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
}


def _validate_labels_plus_minus_one(labels):
    """
    from wood23
    Helper function that ensures that the labels that training/scoring is performed on are (-1, 1), converting from
    (0, 1) if necessary (preserving the positive class as label 1).

    Parameters
    ----------
    labels : ndarray of shape (n_examples)
        labels whose format is to be checked

    Returns
    -------
    labels : ndarray of shape (n_examples)
        Binary labels in the set {-1,1}
    """
    # convert labels from (0, 1) to (1, 1) if necessary
    if np.isin(labels, [0, 1]).all():
        labels += -1 * (labels == 0)
    if not np.isin(labels, [-1, 1]).all():
        raise ValueError
    return labels


def _validate_labels_zero_one(labels):
    """
    from wood23
    Helper function that ensures that the labels that training/scoring is performed on are (0, 1), converting from
    (-1, 1) if necessary (preserving the positive class as label 1).

    labels : ndarray of shape (n_examples)
        labels whose format is to be checked

    Returns
    -------
    labels : ndarray of shape (n_examples)
        Binary labels in the set {0,1}
    """
    if not len(set(labels)) == 2:
        raise ValueError
    if np.isin(labels, [-1, 1]).all():
        labels = 1 * (labels == 1)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError
    return labels


def staged_errors_regr(self):
    # self.etas: (trials, members, points)
    # mean = self.etas.mean(axis=(), keepdims=True)
    # TODO yields garbage
    etas = self.etas
    r = []
    for est_idx  in range(etas.shape[1]):
        if est_idx == 0:
            duals_mean = etas[:, 0, :].mean(axis=1, keepdims=True)
            combined = self._inverse_generator_gradient(duals_mean)
        else:
            # TODO should be est_idx + 1 pretty sure
            m_etas = etas[:, 0:est_idx+1, :]
            duals_mean = m_etas.mean(axis=1, keepdims=True)  # mean over all member etas
            aggregated = self._inverse_generator_gradient(duals_mean) # apply inverse grad op
            combined = aggregated.squeeze(1) # squeeze out members dimension
        error = self._compute_error(combined, self.labels)
        r.append(error)
    rr = np.array(r)  # shape [ens_size, trial, n_points]
    return np.transpose(rr, (1, 0, 2))  # shape [trial, ens_size, n_points]


def staged_errors(self):
    # test errors for each stage
    # for each trial separately
    # self.pred[ ..., k, ...] is prediction of kth estimator
    # TODO so this is evaluated on the test split?
    preds = self.pred
    labels = self.labels
    # in case of test error we have multiple predictions per trial but only one set to evaluate on -- the test data
    # in case of train error we have multiple predictions per trial and also multiple sets to evaluate on -- the individual train datasets
    r = []
    for est_idx in range(preds.shape[1]):
        # combine member predictions
        if est_idx == 0:
            combined = preds[:, 0, :]  # nothing to combine  # note: 0:0 different from 0
        else:
            # TODO should be est_idx + 1 pretty sure
            aggregated = self.aggregator(preds[:, 0:est_idx, :], axis=1)
            combined = aggregated.squeeze(1)
        error = self._compute_error(combined, labels)
        # combined, error should be (3,719)
        r.append(error)
    # r should be list of length 150
    # with elements (3,719)
    rr = np.array(r)  # shape [ens_size, trial, n_points]
    return np.transpose(rr, (1, 0, 2))  # shape [trial, ens_size, n_points]A


def staged_errors_train(self, preds, labels):
    r = []
    for est_idx in range(preds.shape[1]):
        trial_errors = []
        for trial_idx in range(preds.shape[0]):
            if est_idx == 0:
                combined = preds[trial_idx, 0, :]  # nothing to combine  # note: 0:0 different from 0
                combined = np.expand_dims(combined, axis=0)
            else:
                aggregated  = self.aggregator(preds[trial_idx, 0:est_idx, :], axis=0)  # axis=0 because we pick out trial idx
                # combined = aggregated.squeeze(1)
                combined = aggregated
            trial_error = self._compute_error(combined, labels[trial_idx])
            trial_errors.append(trial_error)
        r.append(np.array(trial_errors))
    rr = np.array(r).squeeze(2)  # shape [ens_size, trial, n_points]
    return np.transpose(rr, (1, 0, 2))  # shape [trial, ens_size, n_points]
