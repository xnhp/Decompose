from ruamel.yaml import YAML

import dvc.api
from decompose import dvc_utils
from decompose.dvc_utils import cwd_path, dataset_summary


def main():

    params = dvc.api.params_show("params.yaml")
    args = dvc_utils.parse_args()
    dataset_id = args.dataset

    summary = dataset_summary(dataset_id, params)

    path = cwd_path("plots", "dataset_summary", dataset_id + ".yaml")
    with open(path, 'w') as yaml_file:
        yaml = YAML()
        yaml.dump(summary, yaml_file)


if __name__ == "__main__":
    main()