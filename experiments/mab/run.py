import logging
import os.path
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import openml
import pandas as pd
import yaml
from fedot.api.main import Fedot
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import array_to_input_data
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from golem.core.optimisers.adaptive.experience_buffer import ExperienceBuffer
from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from experiments.fedot_warm_start.run import evaluate_pipeline, get_dataset_ids, \
    get_current_formatted_date, setup_logging, save_experiment_params, timed, COLLECT_METRICS, fit_evaluate_pipeline
from meta_automl.data_preparation.dataset import OpenMLDataset, TabularData
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.datasets_train_test_split import openml_datasets_train_test_split
from meta_automl.data_preparation.file_system import get_project_root, get_cache_dir
from meta_automl.data_preparation.file_system.file_system import get_checkpoints_dir
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.surrogate.data_pipeline_surrogate import PipelineVectorizer
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel

warnings.filterwarnings("ignore")

N_DATASETS = 20
COLLECT_METRICS_ENUM = tuple(map(MetricsRepository.metric_by_id, COLLECT_METRICS))


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


def _drop_datasets_not_in_knowledge_base(dataset_ids: List[int]):
    path_to_knowledge_base = os.path.join(get_project_root(), 'data', 'knowledge_base_0', 'knowledge_base.csv')
    knowledge_base = pd.read_csv(path_to_knowledge_base)

    datasets_to_use = []
    logging.info("Drop datasets that are not in knowledge base...")
    for dataset_id in tqdm(dataset_ids):
        dataset_name = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False,
                                                   download_features_meta_data=False,
                                                   error_if_multiple=True).name
        if dataset_name not in list(set(knowledge_base['dataset_name'].tolist())):
            continue
        datasets_to_use.append(dataset_id)
    return datasets_to_use


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

    # drop datasets that are not in knowledge base
    dataset_ids = _drop_datasets_not_in_knowledge_base(dataset_ids=dataset_ids)

    dataset_ids_train, dataset_ids_test = split_datasets(dataset_ids, N_DATASETS, update_train_test_split=True)
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
    for dataset_id in tqdm(experiment_params_dict['dataset_ids_test'], 'FEDOT, all datasets'):
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
        experiment_date, experiment_date_iso, experiment_date_for_path = get_current_formatted_date()

        save_dir = get_save_dir(experiment_name=experiment_label, dataset=dataset.name, launch_num=str(i))
        print(f'Current launch save dir path: {save_dir}')
        setup_logging(save_dir)
        save_experiment_params(experiment_params_dict, save_dir)
        timeout = config['timeout']
        meta_automl_time = 0

        # get surrogate model
        if experiment_label == 'FEDOT_MAB':
            context_agent_type = config['setups']['FEDOT_MAB']['context_agent_type']
            if context_agent_type == 'surrogate':
                config['setups']['FEDOT_MAB']['context_agent_type'] = _load_pipeline_vectorizer()

        # if pretrained bandit must be used
        if experiment_label == 'FEDOT_MAB':
            adaptive_mutation_type = config['setups']['FEDOT_MAB']['adaptive_mutation_type']
            if adaptive_mutation_type == 'pretrained_contextual_mab':
                # if bandit have not been trained yet for this dataset
                if isinstance(config['setups']['FEDOT_MAB']['adaptive_mutation_type'], str):
                    meta_automl_time = datetime.now().second
                    bandit = _get_pretrained_bandit(dataset=dataset.name,
                                                    datasets_ids_train=experiment_params_dict['dataset_ids_train'],
                                                    use_dataset_encoding=config['setups']['FEDOT_MAB'][
                                                        'use_dataset_encoding'])
                    meta_automl_time = datetime.now().second - meta_automl_time
                    config['setups']['FEDOT_MAB']['adaptive_mutation_type'] = bandit

        # run fedot
        fedot = Fedot(timeout=timeout, logging_level=30, **config['setups'][experiment_label]['launch_params'])
        # test result on test data and save metrics
        fit_func = partial(fedot.fit, features=train_data.x, target=train_data.y)
        result, fit_time = timed(fit_func)()
        evaluate_and_save_results(train_data=train_data,
                                  test_data=test_data,
                                  pipeline=result,
                                  experiment_date=experiment_date,
                                  run_label=f'{launch_num}_{dataset.name}',
                                  save_dir=save_dir,
                                  automl_timeout=timeout * 60,
                                  automl_fit_time=fit_time * 60,
                                  meta_automl_time=meta_automl_time)


def evaluate_and_save_results(train_data: TabularData, test_data: TabularData, pipeline: Pipeline,
                              run_label: str, experiment_date: datetime, save_dir: Path,
                              automl_fit_time: int, automl_timeout: int, meta_automl_time: int):
    train_data_for_fedot = array_to_input_data(train_data.x, train_data.y)
    fit_func = partial(pipeline.fit, train_data_for_fedot)
    evaluate_func = partial(evaluate_pipeline, train_data=train_data, test_data=test_data)
    run_date = datetime.now()
    pipeline, metrics, fit_time = fit_evaluate_pipeline(pipeline=pipeline, fit_func=fit_func,
                                                        evaluate_func=evaluate_func)
    save_evaluation(dataset=train_data.dataset,
                    run_label=run_label,
                    pipeline=pipeline,
                    automl_time_min=automl_fit_time,
                    pipeline_fit_time=fit_time,
                    automl_timeout_min=automl_timeout,
                    meta_learning_time_sec=meta_automl_time,
                    run_data=run_date,
                    experiment_date=experiment_date,
                    save_dir=save_dir,
                    **metrics)
    return pipeline


def save_evaluation(save_dir: Path, dataset, pipeline, **kwargs):
    run_results: Dict[str, Any] = dict(dataset_id=dataset.id,
                                       dataset_name=dataset.name,
                                       model_obj=pipeline,
                                       model_str=pipeline.descriptive_id,
                                       task_type='classification',
                                       **kwargs)
    try:
        histories_dir = save_dir.joinpath('history')
        models_dir = save_dir.joinpath('models')
        eval_results_path = save_dir.joinpath('evaluation_results.csv')

        histories_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)

        dataset_id = run_results['dataset_id']
        run_label = run_results['run_label']
        # define saving paths
        model_path = models_dir.joinpath(f'{dataset_id}_{run_label}')
        history_path = histories_dir.joinpath(f'{dataset_id}_{run_label}_history.json')
        # replace objects with export paths for csv
        run_results['model_path'] = str(model_path)
        run_results.pop('model_obj').save(model_path)
        run_results['history_path'] = str(history_path)
        if 'history_obj' in run_results:
            history_obj = run_results.pop('history_obj')
            if history_obj is not None:
                history_obj.save(run_results['history_path'])

        df_evaluation_properties = pd.DataFrame([run_results])

        if eval_results_path.exists():
            df_results = pd.read_csv(eval_results_path)
            df_results = pd.concat([df_results, df_evaluation_properties])
        else:
            df_results = df_evaluation_properties
        df_results.to_csv(eval_results_path, index=False)

    except Exception as e:
        logging.exception(f'Saving results "{run_results}"')
        if __debug__:
            raise e


def _get_pretrained_bandit(dataset: str, datasets_ids_train: list, use_dataset_encoding: bool = False):
    """ Return pretrained bandit on similar to specified datasets. """

    path_to_knowledge_base = os.path.join(get_project_root(), 'data', 'knowledge_base_0', 'knowledge_base.csv')
    knowledge_base = pd.read_csv(path_to_knowledge_base)
    agent = get_contextual_bandit()

    j = 0
    for dataset_id in datasets_ids_train:
        if j == 2:
            break
        j += 1
        dataset_train = OpenMLDataset(dataset_id)

        dataset_train_name = dataset_train.name
        if dataset_train_name == dataset or dataset_train_name not in list(
                set(knowledge_base['dataset_name'].tolist())):
            continue
        path_to_histories = list(
            set(list(knowledge_base[knowledge_base['dataset_name'] == dataset_train_name]['history_path'])))
        if len(path_to_histories) == 0:
            continue
        agent = train_bandit_on_histories(path_to_histories, agent)
    return agent


def _get_mutation_class(mutation_str: str):
    if 'parameter_change_mutation' in mutation_str:
        return parameter_change_mutation
    if 'MutationTypesEnum' in mutation_str:
        return MutationTypesEnum[mutation_str.split('.')[1]]


def train_bandit_on_histories(path_to_histories: List[str], bandit: ContextualMultiArmedBanditAgent):
    """ Retrieve data from histories and train bandit on it. """
    experience_buffer = ExperienceBuffer()
    for path_to_history in tqdm(path_to_histories):
        history = OptHistory.load(os.path.join(get_project_root(), 'data', 'knowledge_base_0', path_to_history))
        for i, gen in enumerate(history.individuals):
            if i == 2:
                break
            individuals = gen.data
            for ind in individuals:
                # simplify and use only the first operator and parent
                if not ind.parent_operator or \
                        not ind.parent_operator.operators or not ind.parent_operator.parent_individuals:
                    continue

                try:
                    operator = ind.parent_operator.operators[0]
                    # get mutation class since it is stored as str
                    operator = _get_mutation_class(operator)
                    parent_individual = ind.parent_operator.parent_individuals[0]
                    fitness_difference = ind.fitness.value - parent_individual.fitness.value
                    if ind.graph and operator and fitness_difference:
                        experience_buffer.collect_experience(obs=ind, action=operator, reward=fitness_difference)
                # since some individuals have None fitness
                except TypeError:
                    continue

            # check if the is any experience collected
            if experience_buffer._individuals:
                bandit.partial_fit(experience=experience_buffer)
    return bandit


def get_contextual_bandit():
    repo = OperationTypesRepository.assign_repo('model', 'model_repository.json')
    context_agent_type = _load_pipeline_vectorizer()
    mab = ContextualMultiArmedBanditAgent(actions=[parameter_change_mutation,
                                                   MutationTypesEnum.single_change,
                                                   MutationTypesEnum.single_drop,
                                                   MutationTypesEnum.single_add,
                                                   MutationTypesEnum.single_edge],
                                          context_agent_type=context_agent_type,
                                          available_operations=repo.operations)
    return mab


def _load_pipeline_vectorizer() -> PipelineVectorizer:
    """ Loads pipeline vectorizer with surrogate model. """

    surrogate_knowledge_base_dir = get_checkpoints_dir() / 'tabular'

    # Load surrogate model
    surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
        checkpoint_path=surrogate_knowledge_base_dir / "checkpoints/best.ckpt",
        hparams_file=surrogate_knowledge_base_dir / "hparams.yaml"
    )
    surrogate_model.eval()

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
