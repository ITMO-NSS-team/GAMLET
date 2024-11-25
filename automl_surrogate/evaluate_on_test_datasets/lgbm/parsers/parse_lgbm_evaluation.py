"""This module summarizes log file of LGBM evaluations to a csv file."""

import argparse
import re
import warnings

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
        logloss = float(re.findall("Logloss: .+\n", sample)[0].split()[-1])
        rocauc = float(re.findall("ROC-AUC: .+\n", sample)[0].split()[-1])
        res.append(
            {
                "dset": dset,
                "fold": fold,
                "logloss": logloss,
                "rocauc": rocauc,
            }
        )
    pd.DataFrame.from_records(res).to_csv(args.output_csv_file, sep=";", decimal=",")