"""This module contains scripts to evaluate LGBM."""

import argparse
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
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


def train_evaluate(train, test):
    model = TabularPredictor(label="target", eval_metric="log_loss")

    start_time = time.time()
    model.fit(train, num_cpus=1)
    finish_time = time.time()
    elapsed_time = finish_time - start_time
    model_info = model.info()
    (preprocessing_time,) = model_info["time_fit_preprocessing"]
    training_time = model_info["time_fit_training"]
    total_time = model_info["time_fit_total"]

    y_pred_proba = model.predict_proba(test.drop(columns=["target"]))[1]

    y_test = test["target"]
    test_logloss = log_loss(y_test, y_pred_proba)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)

    return test_roc_auc, test_logloss, elapsed_time, preprocessing_time, training_time, total_time


def create_dataset(X, y):
    frame = pd.DataFrame(X)
    frame["target"] = y
    return TabularDataset(frame)


def train_and_eval(datasets_root, logger, dataset_name, fold):
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

    train = create_dataset(train_x, train_y)
    test = create_dataset(test_x, test_y)

    (
        test_roc_auc,
        test_logloss,
        elapsed_time,
        preprocessing_time,
        training_time,
        total_time,
    ) = train_evaluate(train, test)

    logger.info(f"Logloss: {test_logloss:.3f}")
    logger.info(f"ROC-AUC: {test_roc_auc:.3f}")
    logger.info(f"Elapsed time: {elapsed_time:.3f}")
    logger.info(f"Preprocessing time: {preprocessing_time:.3f}")
    logger.info(f"Training time: {training_time:.3f}")
    logger.info(f"Total time: {total_time:.3f}")
    logger.info("-" * 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=str, required=True)
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

    for dataset_name in datasets:
        for fold in range(10):
            train_and_eval(args.datasets_root, logger, dataset_name, fold)
