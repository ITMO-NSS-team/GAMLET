import numpy as np
from pathlib import Path
import pickle
from collections import OrderedDict


def sort_pipelines(selected_graphs_file, dataset_dir, sorted_graphs_file):
    with open(selected_graphs_file, "rb") as f:
        data = pickle.load(f)
    keys = list(data.keys())

    base = Path(dataset_dir)

    lengths = []
    for fname in keys:
        fold_id = int(fname[-1])
        name = fname[:-2]
        file = base.joinpath(f"trainy_{name}_fold{fold_id}.npy")
        lengths.append(len(np.load(file)))

    sorted_data = OrderedDict()
    for i in np.argsort(lengths):
        key = keys[i]
        sorted_data[key] = data[key]

    with open(sorted_graphs_file, "wb") as f:
        pickle.dump(sorted_data, f)

if __name__ == "__main__":
    # DEFINE YOUR PATHS HERE
    sort_pipelines(
        "/home/cherniak/itmo_job/GAMLET/surrogate_hyperparams/collect_dataset/selected_graphs.pickle",
        "/home/cherniak/itmo_job/datasets_folds",
        "/home/cherniak/itmo_job/GAMLET/surrogate_hyperparams/collect_dataset/sorted_selected_graphs.pickle",
    )