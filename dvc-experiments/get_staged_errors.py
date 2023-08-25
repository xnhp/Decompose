"""
Get errors of sub-forest 1..k for k=1..M from saved results
"""
import os

import dvc.api
import numpy as np

from decompose import dvc_utils
from decompose.dvc_utils import parse_args, staged_errors_filepath, results_filepath
from decompose.experiments import load_results


def main():
    args = parse_args()
    model_id = args.model
    dataset_id = args.dataset

    results = load_results(results_filepath(model_id, dataset_id))
    # results = load_results("standard-rf-regressor/results.pkl")

    staged_errors = results.staged_errors
    error_rates = np.mean(staged_errors, axis=2)  # mean over training samples
    # error_rates = np.mean(error_rates, axis=0)  # mean over trials

    r = []
    for trial_idx in range(error_rates.shape[0]):
        for member_idx in range(error_rates.shape[1]):
            r.append({
                'trial_idx': trial_idx,
                'member_idx': member_idx,
                'error': error_rates[trial_idx][member_idx]
            })

    # dvc_utils.json_dump(rr, "dump.json")
    dvc_utils.json_dump(r, staged_errors_filepath(model_id, dataset_id))
    # number of members and errors
    # dvc_utils.dump(error_rates, dvc_utils.staged_errors_filepath())


if __name__ == "__main__":
    main()
