"""This module summarizes log file of LGBM evaluations to a csv file."""

import argparse
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_logfile", type=str, required=True)
    parser.add_argument("--output_csv_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_logfile) as f:
        data = f.read()

    samples = data.split("-" * 10)

    res = []

for sample in samples:
    if sample == "\n":
        continue
    dset_fold = re.findall("Processing .+\n", sample)[0].split()[1].strip()
    dset = dset_fold[:-2]
    fold = dset_fold[-1]
    logloss_vals = list(map(lambda x: float(x.split()[-1]), re.findall("Logloss: .+\n", sample)))
    rocauc_vals = list(map(lambda x: float(x.split()[-1]), re.findall("ROC-AUC: .+\n", sample)))
    res.append(
        {
            "dset": dset,
            "fold": fold,
            "wors_logloss": np.max(logloss_vals),
            "best_logloss": np.min(logloss_vals),
            "mean_logloss": np.mean(logloss_vals),
            "std_logloss": np.std(logloss_vals),
            "median_logloss": np.median(logloss_vals),
            "worst_rocauc": np.min(rocauc_vals),
            "best_rocauc": np.max(rocauc_vals),
            "mean_rocauc": np.mean(rocauc_vals),
            "std_rocauc": np.std(rocauc_vals),
            "median_rocauc": np.median(rocauc_vals),
        }
    )
    columns = ["dset", "fold", "best_rocauc", "best_logloss"]
    pd.DataFrame.from_records(res)[columns].to_csv(args.output_csv_file, sep=";", decimal=",")
