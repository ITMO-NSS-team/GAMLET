import os.path
import timeit
from copy import deepcopy
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
import warnings

from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.surrogate.models import RankingPipelineDatasetSurrogateModel
from thegolem.data_pipeline_surrogate import PipelineVectorizer

warnings.filterwarnings("ignore")


def get_save_dir(dataset: str, experiment_name: str, launch_num: str) -> Path:
    save_dir = get_cache_dir(). \
        joinpath('experiments').joinpath(experiment_name).joinpath(dataset).joinpath(launch_num)
    if not os.path.exists(save_dir):
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
    fedot = Fedot(timeout=timeout, initial_assumption=initial_assumption, **params)
    fedot.fit(x, y)
    automl_time = timeit.default_timer() - time_start

    metrics = evaluate_pipeline(fedot.current_pipeline, fedot.train_data)
    pipeline = fedot.current_pipeline
    run_results = get_result_data_row(dataset=dataset, run_label=run_label, pipeline=pipeline,
                                      automl_time_sec=automl_time, automl_timeout_min=fedot.params.timeout,
                                      history_obj=fedot.history, **metrics)
    return fedot, run_results


def run(path_to_config: str):
    with open(path_to_config, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)

    # Prepare data
    df_datasets_train, df_datasets_test, datasets_dict = fetch_datasets()

    dataset_ids = list(datasets_dict.keys())
    dataset_ids_train = df_datasets_train.index.to_list()
    dataset_ids_test = df_datasets_test.index.to_list()

    dataset_names_train = df_datasets_train['dataset_name'].to_list()
    dataset_names_test = df_datasets_test['dataset_name'].to_list()

    experiment_params_dict = dict(
            input_config=config_dict,
            dataset_ids=dataset_ids,
            dataset_ids_train=dataset_ids_train,
            dataset_names_train=dataset_names_train,
            dataset_ids_test=dataset_ids_test,
            dataset_names_test=dataset_names_test
        )

    experiment_labels = list(config_dict['common_fedot_params'].keys())

    for label in experiment_labels:

        run_experiment(experiment_params_dict=experiment_params_dict,
                       datasets_dict=datasets_dict,
                       experiment_label=label)


def run_experiment(experiment_params_dict: dict, datasets_dict: dict,
                   experiment_label: str):
    for dataset_id, dataset in tqdm(datasets_dict.items(), 'FEDOT, all datasets'):
        experiment_date, experiment_date_iso, experiment_date_for_path = get_current_formatted_date()

        experiment_params_dict['experiment_start_date_iso'] = experiment_date_iso

        run_experiment_per_launch(experiment_params_dict=experiment_params_dict,
                                  experiment_date=experiment_date,
                                  config=deepcopy(experiment_params_dict['input_config']),
                                  dataset_id=dataset_id, dataset=dataset,
                                  experiment_label=experiment_label)


def run_experiment_per_launch(experiment_params_dict, experiment_date, config, dataset_id, dataset,
                              experiment_label):
    best_models_per_dataset = {}
    launch_num = config['launch_num']
    for i in tqdm(range(launch_num)):
        save_dir = get_save_dir(experiment_name=experiment_label, dataset=dataset.name, launch_num=str(i))
        print(f'Current launch save dir path: {save_dir}')
        setup_logging(save_dir)
        save_experiment_params(experiment_params_dict, save_dir)
        timeout = config['timeout']
        run_date = datetime.now()

        # get surrogate model
        if experiment_label == 'FEDOT_MAB':
            context_agent_type = config['common_fedot_params']['FEDOT_MAB']['context_agent_type']
            if context_agent_type == 'surrogate':
                config['common_fedot_params']['FEDOT_MAB']['context_agent_type'] = _load_pipeline_vectorizer()

        fedot, run_results = fit_fedot(dataset=dataset, timeout=timeout, run_label='FEDOT',
                                       **config['common_fedot_params'][experiment_label])
        save_evaluation(run_results, run_date, experiment_date, save_dir)

        # Filter out unique individuals with the best fitness
        history = fedot.history
        best_models = extract_best_models_from_history(dataset, history)
        best_models_per_dataset[dataset_id] = best_models


def _load_pipeline_vectorizer() -> PipelineVectorizer:
    """ Loads pipeline vectorizer with surrogate model. """
    checkpoint_path = os.path.join(get_project_root(), 'experiments', 'base', 'checkpoints', 'last.ckpt')
    hparams_file = os.path.join(get_project_root(), 'experiments', 'base', 'hparams.yaml')
    surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_file
    )

    pipeline_features_extractor = FEDOTPipelineFeaturesExtractor(include_operations_hyperparameters=False,
                                                                 operation_encoding="ordinal")
    pipeline_vectorizer = PipelineVectorizer(
        pipeline_features_extractor=pipeline_features_extractor,
        pipeline_estimator=surrogate_model
    )

    return pipeline_vectorizer


if __name__ == '__main__':
    config_name = 'mab_config.yaml'
    path_to_config = os.path.join(get_project_root(), 'experiments', 'mab_experiment', config_name)
    run(path_to_config=path_to_config)
