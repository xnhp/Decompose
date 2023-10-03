import glob
import logging
import pickle

import numpy as np

from decompose import dvc_utils
from decompose.dvc_utils import cwd_path
from decompose.utils import all_getters


def staged(getter):
    M = 150  # TODO move to params
    ms = range(1, M, 5) # TODO move step to params
    values = np.vectorize(getter)(ms)
    return np.column_stack((ms, values))

def staged_decomp_glob(dataset_path):
    return glob.glob(dataset_path + "/*.npy")


def main():

    args = dvc_utils.parse_args()
    model_id = args.model
    dataset_id = args.dataset

    decomp_path = cwd_path("decomps", dataset_id, model_id + ".pkl")

    with open(decomp_path, "rb") as f:
        try:
            decomp = pickle.load(f)
        except EOFError as e:
            # dont want other stages to fail so log & continue
            logging.error(f"Error loading {decomp_path}: {e}")
            path = cwd_path("staged-decomp-values", dataset_id, model_id, "foo") # still create out path
            return
    for getter_id in all_getters().keys():
        getter = getattr(decomp, getter_id)
        path = cwd_path("staged-decomp-values", dataset_id, model_id, f"{getter_id}.npy")
        np.save(path, staged(getter))

if __name__ == "__main__":
    main()