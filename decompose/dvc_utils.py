import argparse
import logging
import os

from decompose.classifiers import StandardRFClassifier, DRFWeightedRFClassifier, SimpleWeightedRFClassifier
from decompose.regressors import StandardRFRegressor, SquaredErrorGradientRFRegressor


def staged_errors_filepath(model_identifier, dataset_identifier):
    return os.path.join(os.getcwd(),
                        "staged_errors",
                        dataset_identifier,
                        model_identifier + ".json"
                        )


def results_filepath_base():
    return os.path.join(os.getcwd(), "results")

def decomps_filepath(model_identifier, dataset_identifier):
    return cwd_path("decomps",
                    dataset_identifier,
                    model_identifier + "")  # ending attached when saving
def results_filepath(model_identifier, dataset_identifier):
    return os.path.join(os.getcwd(),
                        "results",
                        dataset_identifier,
                        model_identifier + ".pkl")

def cwd_path(*args):
    path = os.path.join(os.getcwd(), *args)
    parents = os.path.dirname(path)
    from pathlib import Path
    Path(parents).mkdir(parents=True, exist_ok=True)
    return path


def json_dump(obj, file_name):
    with open(file_name, "w+", encoding="utf-8") as file_:
        # create parents if not exists
        from pathlib import Path
        import os
        Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
        import json
        json.dump(obj, file_)
        # # dump this object
        # # pickle.dump(obj, file_)
        # np.save(file_name, obj)
        logging.debug(f"Writing results to {file_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False, dest="model", type=str, help="the model identifier")
    parser.add_argument('--dataset', required=False, dest="dataset", type=str, help="the dataset identifier")
    return parser.parse_args()


def get_model(identifier: str):
    models = {
        # regression
        'standard-rf-regressor': StandardRFRegressor(),
        'sqerr-gradient-rf-regressor': SquaredErrorGradientRFRegressor(),
        # classification
        'standard-rf-classifier': StandardRFClassifier(),
        'drf-weighted-classifier': DRFWeightedRFClassifier(),
        'ensemble-weighted-classifier': SimpleWeightedRFClassifier()
    }
    return models[identifier]


def get_n_classes(dataset_id):
    if "mnist" in dataset_id:
        return 10
    if "cover" in dataset_id:
        return 7
    if "qsar-biodeg" in dataset_id:
        return 2
    if "bioresponse" in dataset_id:
        return 2
    else:
        return None
