import os.path
import timeit
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import openml
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from golem.core.optimisers.adaptive.agent_trainer import AgentTrainer
from golem.core.optimisers.adaptive.history_collector import HistoryReader
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.objective import Objective
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

from experiments.fedot_warm_start.run import evaluate_pipeline, save_evaluation, get_dataset_ids, \
    get_current_formatted_date, setup_logging, save_experiment_params
from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.datasets_train_test_split import openml_datasets_train_test_split
from meta_automl.data_preparation.file_system import get_project_root, get_cache_dir
import warnings

from meta_automl.data_preparation.models_loaders.fedot_history_loader import extract_best_models_from_history
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.surrogate.data_pipeline_surrogate import PipelineVectorizer
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel

warnings.filterwarnings("ignore")

N_DATASETS = 20


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


def split_datasets(dataset_ids, n_datasets: Optional[int] = None, update_train_test_split: bool = False) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_path = Path(__file__).parent / 'train_test_datasets_split.csv'

    if update_train_test_split:
        df_split_datasets = openml_datasets_train_test_split(dataset_ids, test_size=0.3, seed=42)
        df_split_datasets.to_csv(split_path)
    else:
        df_split_datasets = pd.read_csv(split_path, index_col=0)

    df_train = df_split_datasets[df_split_datasets['is_train'] == 1]
    df_test = df_split_datasets[df_split_datasets['is_train'] == 0]

    if n_datasets is not None:
        frac = n_datasets / len(df_split_datasets)
        df_train = df_train.sample(frac=frac, random_state=42)
        df_test = df_test.sample(frac=frac, random_state=42)

    datasets_train = df_train.index.to_list()
    datasets_test = df_test.index.to_list()

    return datasets_train, datasets_test


def run(path_to_config: str):
    with open(path_to_config, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)

    dataset_ids = get_dataset_ids()
    dataset_ids_train, dataset_ids_test = split_datasets(dataset_ids, N_DATASETS)
    dataset_ids = dataset_ids_train + dataset_ids_test

    experiment_params_dict = dict(
            input_config=config_dict,
            dataset_ids=dataset_ids,
            dataset_ids_train=dataset_ids_train,
            dataset_ids_test=dataset_ids_test,
        )

    experiment_labels = list(config_dict['setups'].keys())

    # run experiment per setup
    for label in experiment_labels:

        run_experiment(experiment_params_dict=experiment_params_dict,
                       dataset_ids=dataset_ids,
                       experiment_label=label)


def run_experiment(experiment_params_dict: dict, dataset_ids: dict,
                   experiment_label: str):
    dataset_splits = {}
    for dataset_id in tqdm(dataset_ids, 'FEDOT, all datasets'):
        dataset = OpenMLDataset(dataset_id)
        # if dataset.name not in experiment_params_dict['input_config']['datasets']['train']:
        #     continue
        experiment_date, experiment_date_iso, experiment_date_for_path = get_current_formatted_date()

        experiment_params_dict['experiment_start_date_iso'] = experiment_date_iso

        run_experiment_per_launch(experiment_params_dict=experiment_params_dict,
                                  dataset_splits=dataset_splits,
                                  config=deepcopy(experiment_params_dict['input_config']),
                                  dataset_id=dataset_id, dataset=dataset,
                                  dataset_ids=dataset_ids,
                                  experiment_label=experiment_label)


def run_experiment_per_launch(experiment_params_dict, dataset_splits, config, dataset_id, dataset,
                              dataset_ids, experiment_label):
    # get train and test
    dataset_data = dataset.get_data()
    idx_train, idx_test = train_test_split(range(len(dataset_data.y)),
                                           test_size=0.3,
                                           stratify=dataset_data.y,
                                           shuffle=True)
    train_data, test_data = dataset_data[idx_train], dataset_data[idx_test]
    dataset_splits[dataset_id] = dict(train=train_data, test=test_data)

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
            context_agent_type = config['setups']['FEDOT_MAB']['context_agent_type']
            if context_agent_type == 'surrogate':
                config['setups']['FEDOT_MAB']['context_agent_type'] = _load_pipeline_vectorizer()

        # if pretrained bandit is specified
        if experiment_label == 'FEDOT_MAB':
            adaptive_mutation_type = config['setups']['FEDOT_MAB']['adaptive_mutation_type']
            if adaptive_mutation_type == 'pretrained_contextual_mab':
                bandit = _get_pretrained_bandit(dataset=dataset.name, dataset_ids=dataset_ids)
                config['setups']['FEDOT_MAB']['adaptive_mutation_type'] = bandit

        # run fedot
        time_start = timeit.default_timer()
        fedot = Fedot(timeout=timeout, logging_level=30, **config['setups'][experiment_label])
        fedot.fit(train_data)
        automl_time = timeit.default_timer() - time_start

        # test result on test data and save metrics
        metrics = evaluate_pipeline(fedot.current_pipeline, train_data, test_data)
        pipeline = fedot.current_pipeline
        # run_results = get_result_data_row(dataset=dataset, run_label=experiment_label, pipeline=pipeline,
        #                                   automl_time_sec=automl_time, automl_timeout_min=fedot.params.timeout,
        #                                   history_obj=fedot.history, **metrics)

        # save_evaluation(run_results, run_date, experiment_date)

        # Filter out unique individuals with the best fitness
        history = fedot.history
        best_models = extract_best_models_from_history(dataset, history)
        best_models_per_dataset[dataset_id] = best_models


def _get_pretrained_bandit(dataset: str, dataset_ids: list):
    """ Return pretrained bandit on similar to specified datasets. """
    base_path = os.path.join(get_project_root(), 'experiments',
                             'mab_experiment')
    dataset_similaruty_path = os.path.join(base_path, 'dataset_similarity.csv')

    path_to_knowledge_base = os.path.join(base_path, 'knowledge_base.csv')
    knowledge_base = pd.read_csv(path_to_knowledge_base)

    bandit = gather_data_from_histories(path_to_dataset_similarity=dataset_similaruty_path,
                                        datasets=[dataset],
                                        knowledge_base=knowledge_base,
                                        dataset_ids=dataset_ids)[dataset]
    return bandit


def gather_data_from_histories(path_to_dataset_similarity: str, datasets: List[str],
                               knowledge_base, dataset_ids):
    dataset_similarity = pd.read_csv(path_to_dataset_similarity)

    mab_per_dataset = dict.fromkeys(datasets, None)

    for original_dataset_name in datasets:
        similar = dataset_similarity[dataset_similarity['dataset'] == original_dataset_name]['similar_datasets'].tolist()[0]\
            .replace("[", "").replace("]", "").split(" ")

        # for s in similar:
        #     if s != '':
        #         n = int(s)
                # dataset_name = datasets_dict[n].name
                # if dataset_name == original_dataset_name or dataset_name not in list(set(knowledge_base['dataset_name'].tolist())):
                #     continue
                # path_to_histories = list(set(list(knowledge_base[knowledge_base['dataset_name'] == dataset_name]['history_path'])))
                # if len(path_to_histories) == 0:
                #     continue
                # mab_per_dataset[original_dataset_name] = \
                #     train_bandit_on_histories(path_to_histories, get_contextual_bandit())


def pretrain_agent(optimizer: EvoGraphOptimizer, objective: Objective, results_dir: str) -> AgentTrainer:
    agent = optimizer.mutation.agent
    trainer = AgentTrainer(objective, optimizer.mutation, agent)
    # load histories
    history_reader = HistoryReader(Path(results_dir))
    # train agent
    trainer.fit(histories=history_reader.load_histories(), validate_each=1)
    return trainer


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
    path_to_config = os.path.join(get_project_root(), 'experiments', 'mab', config_name)
    run(path_to_config=path_to_config)
