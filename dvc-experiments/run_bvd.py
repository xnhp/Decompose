import os.path
import warnings
import dvc.api
from sklearn.exceptions import DataConversionWarning

from decompose import BVDExperiment, dvc_utils
from decompose.data_utils import load_standard_dataset
from decompose.dvc_utils import get_model

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# TODO fix "Datatype of x is not int, this may give unexpected behaviour"
# e.g. when training stqandard-rf-classifier

"""
    Run a BVD experiment as parameterised by DVC. 
"""


def main():
    args = dvc_utils.parse_args()
    model_id = args.model
    dataset_id = args.dataset
    model = get_model(model_id)
    params = dvc.api.params_show("params.yaml")
    bvd_exp = BVDExperiment(model,
                            **params['bvd_config'],
                            decompositions_filepath=dvc_utils.decomps_filepath(model_id, dataset_id),
                            )
    # mnist dataset comes with own special split
    frac_training = None if dataset_id == "mnist" else params['data']['frac_training']
    dataset = load_standard_dataset(dataset_id, frac_training=frac_training)
    results = bvd_exp.run_experiment(*dataset, n_trials=params['run_experiment_config']['n_trials'], custom_metrics={})
    results.save_results(dvc_utils.results_filepath(model_id, dataset_id))


if __name__ == "__main__":
    main()
