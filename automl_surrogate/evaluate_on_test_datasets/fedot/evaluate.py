"""This module contains scripts to evaluate FEDOT framework w/ and w/o surrogate model, or evaluate a given pipleine."""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import yaml
from fedot.core.pipelines.pipeline import Pipeline
from optimize_with_surrogate import SurrogatePipeline, create_and_test_pipeline

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


def evaluate_fedot(
    logger,
    datasets_dir: str,
    dataset_name: str,
    fold_id: int,
    surrogate: "SurrogatePipeline" = None,
    predefined_models_root: str = None,
):
    train_x_path = os.path.join(datasets_dir, f"train_{dataset_name}_fold{fold}.npy")
    train_y_path = os.path.join(datasets_dir, f"trainy_{dataset_name}_fold{fold}.npy")
    test_x_path = os.path.join(datasets_dir, f"test_{dataset_name}_fold{fold}.npy")
    test_y_path = os.path.join(datasets_dir, f"testy_{dataset_name}_fold{fold}.npy")

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

    if predefined_models_root is not None:
        pipeline_files = list(Path(predefined_models_root).glob("*.json"))

        for i, pipeline_file in enumerate(pipeline_files):
            logger.info(f"Candidate {i}")
            with open(pipeline_file, "r") as f:
                pipeline_json = json.load(f)
            pipeline = Pipeline()
            pipeline = pipeline.load(pipeline_json, dict_fitted_operations=None)
            logger.info(f"{pipeline}")
            create_and_test_pipeline(
                logger,
                f"{dataset_name}_{fold_id}",
                train_x,
                train_y,
                test_x,
                test_y,
                predefined_model=pipeline,
            )
    else:
        create_and_test_pipeline(
            logger,
            f"{dataset_name}_{fold_id}",
            train_x,
            train_y,
            test_x,
            test_y,
            surrogate=surrogate,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=str, required=True)
    parser.add_argument("--logfile", type=str, required=True)
    parser.add_argument("--surrogate_config_file", type=str, required=False, default=None)
    parser.add_argument("--predefined_models_root", type=str, required=False, default=None)
    args = parser.parse_args()

    assert_msg = "Only one of --surrogate_config_file and --predefined_models_root should be provided"
    assert not (args.predefined_models_root is not None and args.surrogate_config_file is True), assert_msg

    logger = logging.getLogger()
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

    if args.surrogate_config_file is not None:
        logger.info("Evaluate with surrogate")
        with open(args.surrogate_config_file, "r") as f:
            surrogate_config = yaml.safe_load(f)
        surrogate = SurrogatePipeline(**surrogate_config)
    else:
        logger.info("Evaluate without surrogate")

    for fold in range(10):
        for dataset_name in datasets:
            evaluate_fedot(
                logger,
                args.datasets_root,
                dataset_name,
                fold,
                surrogate,
                args.predefined_models_root,
            )
