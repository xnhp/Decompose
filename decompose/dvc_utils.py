import argparse
import logging
import os

from fuzzywuzzy import fuzz

from decompose.classifiers import StandardRFClassifier, DRFWeightedBootstrapRFClassifier, SimpleWeightedRFClassifier, \
    DRFWeightedFitRFClassifier, DRFWeightedFitOOBRFClassifier, make_geometric_nn_ensemble, MLP
from decompose.data_utils import load_standard_dataset
from decompose.regressors import StandardRFRegressor, SqErrBoostedBase, SqErrBoostedShallow, SqErrBoostedNoBootstrap, \
    StandardRFNoBootstrap, SqErrBoostedClipped


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
        'standard-rf-nobootstrap': StandardRFNoBootstrap(),
        'sqerr-boosted-shallow': SqErrBoostedShallow(),
        'sqerr-boosted-nobootstrap': SqErrBoostedNoBootstrap(),
        'sqerr-boosted-clipped': SqErrBoostedClipped(),

        # classification
        # random forests
        'standard-rf-classifier': StandardRFClassifier(),
        'drf-weighted-bootstrap-classifier': DRFWeightedBootstrapRFClassifier(),
        'drf-weighted-fit-classifier': DRFWeightedFitRFClassifier(),
        'drf-weighted-fit-oob-classifier': DRFWeightedFitOOBRFClassifier(),
        'ensemble-weighted-classifier': SimpleWeightedRFClassifier(),
        # neural networks
        'ce-nn': make_geometric_nn_ensemble(MLP())
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
    if "spambase-openml" in dataset_id:
        return 2
    if "diabetes" in dataset_id:
        return 2
    if "digits" in dataset_id:
        return 10
    else:
        return None


# TODO would be good if these were distinct from get_fn_color -- choose some colormap?
def get_model_color(identifier: str):
    colors = {
        # regression
        'standard-rf-regressor': "blue",
        'standard-rf-nobootstrap': "yellow",
        'sqerr-boosted-shallow': "green",
        'sqerr-boosted-nobootstrap': "red",
        'sqerr-boosted-clipped': "orange",
        # classification
        'standard-rf-classifier': "blue",
        'drf-weighted-bootstrap-classifier': "red",
        'drf-weighted-fit-classifier': "orange",
        'drf-weighted-fit-oob-classifier': "brown",
        'ensemble-weighted-classifier': "green",
        'ce-nn': "blue"
    }
    return colors[identifier]

def get_fn_color(identifier:str):
    colors = {
        'ambiguity': "green",
        'random_model_threshold': "red",

        'ensemble-error': "gray",
        "ensemble loss": "gray",

        "ensemble bias": "red",
        "avg bias": "orange",

        "variance": "green",
        "variance-effect": "green",

        "diversity": "blue",
        "diversity-effect": "blue",

        'avg-member-loss': "blue",

        'margin': "green"
    }
    return match_dict_key(colors, identifier)

def match_dict_key(dict, identifier):
    return dict[match(identifier, dict.keys())]

def match(query_string, string_list):
    best_match = None
    best_score = 0
    for string in string_list:
        similarity_score = fuzz.ratio(query_string, string)

        # Update best match if the current score is higher
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = string

    if best_score < 0.9:
        return None
    return best_match


def dataset_summary(dataset_id, params=None):
    frac_training = None if dataset_id == "mnist" or params is None else params['data']['frac_training']
    if frac_training is None:
        frac_training = 0.75 # TODO see params.yaml
    train_data, train_labels, test_data, test_labels = load_standard_dataset(dataset_id, frac_training=frac_training)
    summary = {
        "n_classes": get_n_classes(dataset_id),
        "n_train": train_data.shape[0],
        "n_test": test_data.shape[0],
        "dimensions": train_data.shape[1],
        "frac_training": frac_training
    }
    return summary
