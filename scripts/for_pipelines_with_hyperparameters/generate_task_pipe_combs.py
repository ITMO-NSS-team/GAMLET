"This script generate task_pipe_comb file for dataset with pipeline with different hyperparameters."

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# DEFINE YOUR PATHS HERE
dataset_dir = "/Users/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines__no_type_with_hyperparams(4_with_learnables)"

dataset_dir = Path(dataset_dir)
records_file = dataset_dir.joinpath("records.pickle")
id2dataset_file = dataset_dir.joinpath("id2dataset_id.pickle")
with open(records_file, "rb") as f:
    records = pickle.load(f)

with open(id2dataset_file, "rb") as f:
    id2dataset = pickle.load(f)
dataset2id = {v: k for k, v in id2dataset.items()}

train_datasets = os.listdir(dataset_dir.joinpath("train"))
test_datasets = os.listdir(dataset_dir.joinpath("test"))

# Filter MacOS system files.
train_ids = [dataset2id[ds] for ds in train_datasets if ds != ".DS_Store"]
test_ids = [dataset2id[ds] for ds in test_datasets if ds != ".DS_Store"]

df = pd.DataFrame.from_records(records)

# Check if data is OK.
print((df == np.inf).sum())
print()
print(df.isna().sum())

# Remove broken samples.
df = df[df.metric != np.inf]

train_df = df[df.task_id.apply(lambda x: x in train_ids)]
test_df = df[df.task_id.apply(lambda x: x in test_ids)]

print(train_df.shape, test_df.shape)

train_df.to_csv(dataset_dir.joinpath("train_task_pipe_comb.csv"))
test_df.to_csv(dataset_dir.joinpath("test_task_pipe_comb.csv"))
