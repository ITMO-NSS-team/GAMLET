"""Script to compare already collected data with data to be collected."""

import pickle
from pathlib import Path
import pandas as pd


def main(collected_samples_dir, file_with_pipelines_to_be_processed):
    files = list(Path(collected_samples_dir).glob("**/*.pickle"))

    pipe_names = []
    dataset_names = []
    for e in files:
        dataset_names.append(e.parent.name)
        pipe_names.append(e.name.split(".")[0])

    df1 = pd.DataFrame()
    df1["dataset_names"] = dataset_names
    df1["pipe_names"] = pipe_names

    with open(file_with_pipelines_to_be_processed, "rb") as f:
        selected = pickle.load(f)

    pipe_names = []
    dataset_names = []
    for k, v in selected.items():
        for e in v:
            p = Path(e)
            dataset_names.append(k)
            pipe_names.append(p.name.split(".")[0])

    df2 = pd.DataFrame()
    df2["dataset_names"] = dataset_names
    df2["pipe_names"] = pipe_names

    print(f"Total collected sample {len(df1)}")
    print(f"Total samples to be collected {len(df2)}")
    print()

    l1 = df1.groupby("dataset_names").apply(len)
    l2 = df2.groupby("dataset_names").apply(len)
    for name in l2.index:
        if name not in l1.index:
            print(f"{name=}, selected: {l2.loc[name]}, ready: 0")
        elif l2.loc[name] != l1.loc[name]:
            print(f"{name=}, selected: {l2.loc[name]}, ready: {l1.loc[name]}")

if __name__ == "__main__":
    # DEFINE YOUR PATH HERE
    collected_samples_dir ="/home/cherniak/itmo_job/graphs_with_hyperparameters"
    file_with_pipelines_to_be_processed = "/home/cherniak/itmo_job/GAMLET/surrogate_hyperparams/collect_dataset/selected_graphs.pickle"
    main(collected_samples_dir, file_with_pipelines_to_be_processed)