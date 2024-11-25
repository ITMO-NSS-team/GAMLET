"""This script downloads evaluation datasets and split them into 10 cross-validation folds."""

import argparse
import logging
import os

import numpy as np
import openml
from sklearn.model_selection import KFold

datasets = [
    "numerai28_6",  # Will be fixed to correct OpenML name.
    "sf-police-incidents",
    "airlines",
    "albert",
    "kddcup09_appetency",
    "christine",
    "guillermo",
    "amazon_employee_access",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, required=True)
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not os.path.exists(args.datasets_dir):
        os.makedirs(args.datasets_dir)

    for dataset_name in datasets:
        logger.info(f"Processing dataset {dataset_name}")
        if dataset_name == "numerai28_6":
            fixed_dataset_name = "numerai28.6"
        else:
            fixed_dataset_name = dataset_name
        dataset = openml.datasets.get_dataset(fixed_dataset_name, download_data=True)
        x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        x = x.to_numpy()
        y = y.to_numpy()
        kfold = KFold(n_splits=10, random_state=1, shuffle=True)
        for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
            logger.info(f"Processing fold {fold}")
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
            np.save(os.path.join(args.datasets_dir, f"train_{dataset_name}_fold{fold}.npy"), x_train)
            np.save(os.path.join(args.datasets_dir, f"trainy_{dataset_name}_fold{fold}.npy"), y_train)
            np.save(os.path.join(args.datasets_dir, f"test_{dataset_name}_fold{fold}.npy"), x_test)
            np.save(os.path.join(args.datasets_dir, f"testy_{dataset_name}_fold{fold}.npy"), y_test)
