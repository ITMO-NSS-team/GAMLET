"""This module contains scripts to evaluate LGBM with best configurations in the training set."""

import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")

# Datasets names to be used in evaluation.
datasets = [
    "numerai28_6",
    "sf-police-incidents",
    "airlines",
    "albert",
    "kddcup09_appetency",
    "christine",
    "guillermo",
    "amazon_employee_access",
]


def train_and_eval(datasets_root, kwargs_df, logger, dataset_name, fold):
    train_x_path = os.path.join(datasets_root, f"train_{dataset_name}_fold{fold}.npy")
    train_y_path = os.path.join(datasets_root, f"trainy_{dataset_name}_fold{fold}.npy")
    test_x_path = os.path.join(datasets_root, f"test_{dataset_name}_fold{fold}.npy")
    test_y_path = os.path.join(datasets_root, f"testy_{dataset_name}_fold{fold}.npy")

    paths = [train_x_path, train_y_path, test_x_path, test_y_path]
    for p in paths:
        if not os.path.exists(p):
            logger.info(f"Dataset {dataset_name}_{fold} path does not exists. Skip iteration.")
            return

    logger.info(f"Processing {dataset_name}_{fold}")
    logger.info(f"{train_x_path=}")
    logger.info(f"{train_y_path=}")
    logger.info(f"{test_x_path=}")
    logger.info(f"{test_y_path=}")

    train_x = np.load(train_x_path)
    train_y = np.load(train_y_path)
    test_x = np.load(test_x_path)
    test_y = np.load(test_y_path)

    for i, row in kwargs_df.iterrows():
        logger.info(f"Candidate {i}")
        kwargs = row.to_dict()
        if not isinstance(kwargs["class_weight"], str):
            kwargs["class_weight"] = None
        logger.info(f"{kwargs=}")
        model = LGBMClassifier(**kwargs, verbose=-1)
        model.fit(train_x, train_y)
        y_pred_proba = model.predict_proba(test_x)[:, 1]
        test_logloss = log_loss(test_y, y_pred_proba)
        test_roc_auc = roc_auc_score(test_y, y_pred_proba)
        logger.info(f"Logloss: {test_logloss:.3f}")
        logger.info(f"ROC-AUC: {test_roc_auc:.3f}")
        logger.info("")
    logger.info("-" * 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=str, required=True)
    parser.add_argument("--configurations_csv", type=str, required=True)
    parser.add_argument("--logfile", type=str, required=True)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(args.logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    kwargs_df = pd.read_csv(args.configurations_csv)
    hparams_cols = [
        "class_weight",
        "num_leaves",
        "learning_rate",
        "colsample_bytree",
        "subsample",
        "reg_alpha",
        "reg_lambda",
    ]
    kwargs_df = kwargs_df[hparams_cols]

    for dataset_name in datasets:
        for fold in range(10):
            train_and_eval(args.datasets_root, kwargs_df, logger, dataset_name, fold)
