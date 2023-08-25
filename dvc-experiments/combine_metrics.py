import glob
import json
import os.path

from decompose.dvc_utils import parse_args


def main():
    args = parse_args()

    combined_metrics = []

    for filepath in glob.iglob(f"staged_errors/{args.dataset}/*.json"):
        with open(filepath) as f:
            metrics = json.load(f)

            name = os.path.basename(filepath)
            name = os.path.splitext(name)[0]  # filename without suffix

            for obj in metrics:
                obj['model'] = name

            # metrics["model"] = name  # Add a new field to distinguish the models
            combined_metrics += metrics
    with open(f"staged_errors/{args.dataset}-combined.json", "w") as f:
        json.dump(combined_metrics, f)



# import os
# def get_immediate_subdirectories(a_dir):
#     return [name for name in os.listdir(a_dir)
#             if os.path.isdir(os.path.join(a_dir, name))]



if __name__ == "__main__":
    main()
