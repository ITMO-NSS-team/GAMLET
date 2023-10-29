"""This module summarizes log file of FEDOT evaluations w/ or w/o surrogate to a csv file."""

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

    samples = data.split("Processing")[1:]

    res = []

    for sample in samples:
        dset_fold = sample.split("\n")[0].strip()
        dset = dset_fold[:-2]
        fold = dset_fold[-1]
        metrics = re.findall("Prediction on test: \{.+\}", sample)
        if len(metrics) == 0:
            continue
        metrics = metrics[0]
        metrics = re.findall("\{.+\}", metrics)[0]
        metrics = eval(metrics)
        time = re.findall("Elapsed time \d+.\d+", sample)
        if len(time) < 0:
            continue
        time = float(time[0].split()[-1])
        res.append(
            {
                "dset": dset,
                "fold": fold,
                "logloss": metrics["neg_log_loss"],
                "roc_auc": metrics["roc_auc"],
                "time": time,
            }
        )

    pd.DataFrame.from_records(res).to_csv(args.output_csv_file, sep=";", decimal=",")
