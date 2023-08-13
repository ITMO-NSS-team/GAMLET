import logging
import os.path
import timeit
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import openml
import pandas as pd
from fedot.api.main import Fedot
from tqdm import tqdm
import yaml

from experiments.fedot_warm_start.run import get_current_formatted_date, setup_logging, \
    save_experiment_params, extract_best_models_from_history, save_evaluation, \
    get_result_data_row, evaluate_pipeline, transform_data_for_fedot
from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.datasets_train_test_split import openml_datasets_train_test_split
from meta_automl.data_preparation.file_system import get_project_root, get_cache_dir


def get_save_dir(time_now_for_path) -> Path:
    save_dir = get_cache_dir(). \
        joinpath('experiments').joinpath('mab_experiment').joinpath(f'run_{time_now_for_path}')
    save_dir.mkdir(parents=True)
    return save_dir


def fetch_datasets(n_datasets: Optional[int] = None, seed: int = 42, test_size: float = 0.25) \
        -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, OpenMLDataset]]:
    """Returns dictionary with dataset names and cached datasets downloaded from OpenML."""

    dataset_ids = openml.study.get_suite(99).data
    if n_datasets is not None:
        dataset_ids = pd.Series(dataset_ids)
        dataset_ids = dataset_ids.sample(n=n_datasets, random_state=seed)

    df_split_datasets = openml_datasets_train_test_split(dataset_ids, test_size=test_size, seed=seed)
    df_datasets_train = df_split_datasets[df_split_datasets['is_train'] == 1]
    df_datasets_test = df_split_datasets[df_split_datasets['is_train'] == 0]

    datasets = {dataset.id_: dataset for dataset in OpenMLDatasetsLoader().load(dataset_ids)}
    return df_datasets_train, df_datasets_test, datasets


def fit_fedot(dataset: OpenMLDataset, timeout: float, run_label: str, initial_assumption=None, **params) \
        -> (Fedot, Dict[str, Any]):
    """ Runs Fedot evaluation on the dataset, the evaluates the final pipeline on the dataset.
     Returns Fedot instance & properties of the run along with the evaluated metrics. """
    x, y = transform_data_for_fedot(dataset.get_data())

    time_start = timeit.default_timer()
    # TODO: specify agent and surrogate_model
    fedot = Fedot(timeout=timeout, initial_assumption=initial_assumption, **params)
    fedot.fit(x, y)
    automl_time = timeit.default_timer() - time_start

    metrics = evaluate_pipeline(fedot.current_pipeline, fedot.train_data)
    pipeline = fedot.current_pipeline
    run_results = get_result_data_row(dataset=dataset, run_label=run_label, pipeline=pipeline,
                                      automl_time_sec=automl_time, automl_timeout_min=fedot.params.timeout,
                                      history_obj=fedot.history, **metrics)
    return fedot, run_results


def run_experiment(path_to_config: str):
    with open(path_to_config, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)

    experiment_date, experiment_date_iso, experiment_date_for_path = get_current_formatted_date()
    save_dir = get_save_dir(experiment_date_for_path)
    setup_logging(save_dir)
    progress_file_path = save_dir.joinpath('progress.txt')

    df_datasets_train, df_datasets_test, datasets_dict = fetch_datasets()

    dataset_ids = list(datasets_dict.keys())
    dataset_ids_train = df_datasets_train.index.to_list()
    dataset_ids_test = df_datasets_test.index.to_list()

    dataset_names_train = df_datasets_train['dataset_name'].to_list()
    dataset_names_test = df_datasets_test['dataset_name'].to_list()

    datasets_dict_test = dict(filter(lambda item: item[0] in dataset_ids_test, datasets_dict.items()))

    experiment_params_dict = dict(
            experiment_start_date_iso=experiment_date_iso,
            input_config=config_dict,
            dataset_ids=dataset_ids,
            dataset_ids_train=dataset_ids_train,
            dataset_names_train=dataset_names_train,
            dataset_ids_test=dataset_ids_test,
            dataset_names_test=dataset_names_test,
        )
    save_experiment_params(experiment_params_dict, save_dir)

    # run_classic_fedot(progress_file_path=progress_file_path, datasets_dict=datasets_dict,
    #                   experiment_date=experiment_date, save_dir=save_dir,
    #                   config=config_dict)

    run_mab_fedot(progress_file_path=progress_file_path, datasets_dict=datasets_dict,
                  experiment_date=experiment_date, save_dir=save_dir,
                  config=config_dict)


def run_classic_fedot(progress_file_path, datasets_dict, experiment_date, save_dir, config):
    best_models_per_dataset = {}
    launch_num = config['launch_num']
    with open(progress_file_path, 'a') as progress_file:
        for i in range(launch_num):
            for dataset_id, dataset in tqdm(datasets_dict.items(), 'FEDOT, all datasets', file=progress_file):
                try:
                    timeout = config['timeout']
                    run_date = datetime.now()
                    fedot, run_results = fit_fedot(dataset=dataset, timeout=timeout, run_label='FEDOT',
                                                   **config['common_fedot_params'])
                    save_evaluation(run_results, run_date, experiment_date, save_dir)

                    # Filter out unique individuals with the best fitness
                    history = fedot.history
                    best_models = extract_best_models_from_history(dataset, history)
                    best_models_per_dataset[dataset_id] = best_models
                except Exception:
                    logging.exception(f'Train dataset "{dataset_id}"')


def run_mab_fedot(progress_file_path, datasets_dict, experiment_date, save_dir, config):
    best_models_per_dataset = {}
    launch_num = config['launch_num']
    with open(progress_file_path, 'a') as progress_file:
        for i in range(launch_num):
            for dataset_id, dataset in tqdm(datasets_dict.items(), 'FEDOT, all datasets', file=progress_file):
                try:
                    timeout = config['timeout']
                    agent = config['agent']
                    run_date = datetime.now()
                    config['common_fedot_params']['agent'] = agent
                    fedot, run_results = fit_fedot(dataset=dataset, timeout=timeout, run_label='MAB_FEDOT',
                                                   **config['common_fedot_params'])
                    save_evaluation(run_results, run_date, experiment_date, save_dir)

                    # Filter out unique individuals with the best fitness
                    history = fedot.history
                    best_models = extract_best_models_from_history(dataset, history)
                    best_models_per_dataset[dataset_id] = best_models
                except Exception:
                    logging.exception(f'Train dataset "{dataset_id}"')


if __name__ == '__main__':
    config_name = 'mab_config.yaml'
    path_to_config = os.path.join(get_project_root(), 'experiments', 'mab_experiment', config_name)
    run_experiment(path_to_config=path_to_config)
