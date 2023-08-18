import os.path
import timeit
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import openml
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

from experiments.fedot_warm_start.run import get_current_formatted_date, setup_logging, \
    save_experiment_params, extract_best_models_from_history, save_evaluation, \
    get_result_data_row, evaluate_pipeline, transform_data_for_fedot
from experiments.mab_experiment.gather_data_from_histories import gather_data_from_histories
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


def run(path_to_config: str):
    with open(path_to_config, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)

    # Prepare data
    df_datasets_train, df_datasets_test, datasets_dict = fetch_datasets()

    dataset_ids = list(datasets_dict.keys())
    dataset_ids_train = df_datasets_train.index.to_list()

    dataset_names_train = df_datasets_train['dataset_name'].to_list()

    experiment_params_dict = dict(
            input_config=config_dict,
            dataset_ids=dataset_ids,
            dataset_ids_train=dataset_ids_train,
            dataset_names_train=dataset_names_train,
        )

    experiment_labels = list(config_dict['common_fedot_params'].keys())

    for label in experiment_labels:

        run_experiment(experiment_params_dict=experiment_params_dict,
                       datasets_dict=datasets_dict,
                       experiment_label=label)


def run_experiment(experiment_params_dict: dict, datasets_dict: dict,
                   experiment_label: str):
    for dataset_id, dataset in tqdm(datasets_dict.items(), 'FEDOT, all datasets'):
        if dataset.name not in experiment_params_dict['input_config']['datasets']:
            continue
        experiment_date, experiment_date_iso, experiment_date_for_path = get_current_formatted_date()

        experiment_params_dict['experiment_start_date_iso'] = experiment_date_iso

        run_experiment_per_launch(experiment_params_dict=experiment_params_dict,
                                  experiment_date=experiment_date,
                                  config=deepcopy(experiment_params_dict['input_config']),
                                  dataset_id=dataset_id, dataset=dataset,
                                  datasets_dict=datasets_dict,
                                  experiment_label=experiment_label)


def run_experiment_per_launch(experiment_params_dict, experiment_date, config, dataset_id, dataset,
                              datasets_dict, experiment_label):

    train_data, test_data = _split_data_train_test(dataset=dataset)

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

        # if pretrained bandit is specified
        if experiment_label == 'FEDOT_MAB':
            adaptive_mutation_type = config['common_fedot_params']['FEDOT_MAB']['adaptive_mutation_type']
            if adaptive_mutation_type == 'pretrained_contextual_mab':
                bandit = _get_pretrained_bandit(dataset=dataset.name, datasets_dict=datasets_dict)
                config['common_fedot_params']['FEDOT_MAB']['adaptive_mutation_type'] = bandit

        # run fedot
        time_start = timeit.default_timer()
        fedot = Fedot(timeout=timeout, logging_level=30, **config['common_fedot_params'][experiment_label])
        fedot.fit(train_data)
        automl_time = timeit.default_timer() - time_start

        # test result on test data and save metrics
        metrics = evaluate_pipeline(fedot.current_pipeline, test_data)
        pipeline = fedot.current_pipeline
        run_results = get_result_data_row(dataset=dataset, run_label=experiment_label, pipeline=pipeline,
                                          automl_time_sec=automl_time, automl_timeout_min=fedot.params.timeout,
                                          history_obj=fedot.history, **metrics)

        save_evaluation(run_results, run_date, experiment_date, save_dir)

        # Filter out unique individuals with the best fitness
        history = fedot.history
        best_models = extract_best_models_from_history(dataset, history)
        best_models_per_dataset[dataset_id] = best_models


def _get_pretrained_bandit(dataset: str, datasets_dict: dict):
    """ Return pretrained bandit on similar to specified datasets. """
    base_path = os.path.join(get_project_root(), 'experiments',
                             'mab_experiment')
    dataset_similaruty_path = os.path.join(base_path, 'dataset_similarity.csv')

    path_to_knowledge_base = os.path.join(base_path, 'knowledge_base.csv')
    knowledge_base = pd.read_csv(path_to_knowledge_base)

    bandit = gather_data_from_histories(path_to_dataset_similarity=dataset_similaruty_path,
                                        datasets=[dataset],
                                        knowledge_base=knowledge_base,
                                        datasets_dict=datasets_dict)[dataset]
    return bandit


def _split_data_train_test(dataset: OpenMLDataset, seed: int = 42) -> Tuple[InputData, InputData]:
    """ OpenMLDataset -> InputData """
    x, y = transform_data_for_fedot(dataset.get_data())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed,
                                                        stratify=y)

    train_data = InputData(idx=np.arange(0, len(x_train)), features=x_train,
                           target=y_train, task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)

    test_data = InputData(idx=np.arange(0, len(x_test)), features=x_test,
                          target=y_test, task=Task(TaskTypesEnum.classification),
                          data_type=DataTypesEnum.table)

    return train_data, test_data


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
