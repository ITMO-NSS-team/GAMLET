"""This module summarizes log file of best pipeline evaluations to a csv file."""

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

    samples = data.split("Processing")[1:]

    res = []

    for sample in samples:
        dset_fold = sample.split("\n")[0].strip()
        dset = dset_fold[:-2]
        fold = dset_fold[-1]
        metrics = re.findall("Prediction on test: \{.+\}", sample)

        if len(metrics) == 0:
            continue
        logloss = []
        roc_auc = []
        for metrics_ in metrics:
            metrics = re.findall("\{.+\}", metrics_)[0]
            metrics = eval(metrics)
            logloss.append(metrics["neg_log_loss"])
            roc_auc.append(metrics["roc_auc"])
        logloss = np.mean(logloss)
        roc_auc = np.mean(roc_auc)
        res.append(
            {
                "dset": dset,
                "fold": fold,
                "logloss": logloss,
                "roc_auc": roc_auc,
            }
        )

    pd.DataFrame.from_records(res).to_csv(args.output_csv_file, sep=";", decimal=",")
