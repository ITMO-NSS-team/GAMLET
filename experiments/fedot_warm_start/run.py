from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import timeit
from datetime import datetime
from functools import partial, wraps, reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

import loguru
import openml
import pandas as pd
import yaml
from fedot.api.main import Fedot
from fedot.core.data.data import array_to_input_data
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.metrics_repository import MetricsRepository, QualityMetricsEnum
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from pecapiku import CacheDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing_extensions import Literal

from gamlet.approaches.knn_similarity_model_advice import KNNSimilarityModelAdvice
from gamlet.data_preparation.dataset import DatasetIDType, OpenMLDataset, TabularData
from gamlet.data_preparation.datasets_train_test_split import openml_datasets_train_test_split
from gamlet.data_preparation.file_system import get_cache_dir

CONFIGS_DIR = Path(__file__).parent

with open(CONFIGS_DIR / 'configs_list.yaml', 'r') as config_file:
    configs_list = yaml.load(config_file, yaml.Loader)

config = {}
for conf_name in configs_list:
    with open(CONFIGS_DIR / conf_name, 'r') as config_file:
        conf = yaml.load(config_file, yaml.Loader)
    intersection = set(config).intersection(set(conf))
    if intersection:
        raise ValueError(f'Parameter values given twice: {conf_name}, {intersection}.')
    config.update(conf)

# Load constants
SEED = config['seed']
N_DATASETS = config['n_datasets']
TEST_SIZE = config['test_size']
TRAIN_TIMEOUT = config['train_timeout']
TEST_TIMEOUT = config['test_timeout']
N_BEST_DATASET_MODELS_TO_MEMORIZE = config['n_best_dataset_models_to_memorize']
ASSESSOR_PARAMS = config['assessor_params']
ADVISOR_PARAMS = config['advisor_params']
MF_EXTRACTOR_PARAMS = config['mf_extractor_params']
COLLECT_METRICS = config['collect_metrics']
FEDOT_PARAMS = config['fedot_params']
DATA_TEST_SIZE = config['data_test_size']
DATA_SPLIT_SEED = config['data_split_seed']
BASELINE_MODEL = config['baseline_model']
N_AUTOML_REPETITIONS = config['n_automl_repetitions']
# Optional values
TMPDIR = config.get('tmpdir')
SAVE_DIR_PREFIX = config.get('save_dir_prefix')

UPDATE_TRAIN_TEST_DATASETS_SPLIT = config.get('update_train_test_datasets_split')

# Postprocess constants
COLLECT_METRICS_ENUM = tuple(map(MetricsRepository.get_metric, COLLECT_METRICS))
COLLECT_METRICS[COLLECT_METRICS.index('neg_log_loss')] = 'logloss'


def setup_logging(save_dir: Path):
    """ Creates "log.txt" at the "save_dir" and redirects all logging output to it. """
    loguru.logger.add(save_dir / "file_{time}.log")
    log_file = save_dir.joinpath('log.txt')
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        force=True,
        level=logging.DEBUG,
    )


def get_current_formatted_date() -> Tuple[datetime, str, str]:
    """ Returns current date in the following formats:

        1. datetime
        2. str: ISO
        3. str: ISO compatible with Windows file system path (with "." instead of ":") """
    time_now = datetime.now()
    time_now_iso = time_now.isoformat(timespec="minutes")
    time_now_for_path = time_now_iso.replace(":", ".")
    return time_now, time_now_iso, time_now_for_path


def get_save_dir(time_now_for_path) -> Path:
    save_dir = get_cache_dir(). \
        joinpath('experiments').joinpath('fedot_warm_start').joinpath(f'run_{time_now_for_path}')
    if SAVE_DIR_PREFIX:
        save_dir = save_dir.with_name(SAVE_DIR_PREFIX + save_dir.name)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)

    return save_dir


def get_dataset_ids() -> List[DatasetIDType]:
    dataset_ids = openml.study.get_suite(99).data
    if N_DATASETS is not None:
        dataset_ids = pd.Series(dataset_ids)
        dataset_ids = dataset_ids.sample(n=N_DATASETS, random_state=SEED)
    return list(dataset_ids)


def split_datasets(dataset_ids, n_datasets: Optional[int] = None, update_train_test_split: bool = False) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_path = Path(__file__).parent / 'train_test_datasets_split.csv'

    if update_train_test_split:
        df_split_datasets = openml_datasets_train_test_split(dataset_ids, test_size=TEST_SIZE, seed=SEED)
        df_split_datasets.to_csv(split_path)
    else:
        df_split_datasets = pd.read_csv(split_path, index_col=0)

    df_train = df_split_datasets[df_split_datasets['is_train'] == 1]
    df_test = df_split_datasets[df_split_datasets['is_train'] == 0]

    if n_datasets is not None:
        frac = n_datasets / len(df_split_datasets)
        df_train = df_train.sample(frac=frac, random_state=SEED)
        df_test = df_test.sample(frac=frac, random_state=SEED)

    datasets_train = df_train.index.to_list()
    datasets_test = df_test.index.to_list()

    return datasets_train, datasets_test


def evaluate_pipeline(pipeline: Pipeline,
                      train_data: TabularData,
                      test_data: TabularData,
                      metrics: Sequence[QualityMetricsEnum] = COLLECT_METRICS_ENUM,
                      metric_names: Sequence[str] = COLLECT_METRICS,
                      mode: Literal['fitness', 'float'] = 'float'
                      ) -> Union[Dict[str, float], Tuple[Fitness, Sequence[str]]]:
    """Gets quality metrics for the fitted pipeline.
    The function is based on `Fedot.get_metrics()`

    Returns:
        the values of quality metrics
    """
    train_data = array_to_input_data(train_data.x, train_data.y)
    test_data = array_to_input_data(test_data.x, test_data.y)

    def data_producer():
        yield train_data, test_data

    objective = MetricsObjective(metrics)
    obj_eval = PipelineObjectiveEvaluate(objective=objective,
                                         data_producer=data_producer,
                                         eval_n_jobs=-1)

    fitness = obj_eval.evaluate(pipeline)
    if mode == 'float':
        metric_values = fitness.values
        metric_values = {metric_name: round(value, 3) for (metric_name, value) in zip(metric_names, metric_values)}
        return metric_values
    if mode == 'fitness':
        return fitness, metric_names


def timed(func, resolution: Literal['sec', 'min'] = 'min'):
    @wraps(func)
    def wrapper(*args, **kwargs):
        time_start = timeit.default_timer()
        result = func(*args, **kwargs)
        time_delta = timeit.default_timer() - time_start
        if resolution == 'min':
            time_delta /= 60
        return result, time_delta

    return wrapper


def fit_evaluate_automl(fit_func, evaluate_func) -> (Fedot, Dict[str, Any]):
    """ Runs Fedot evaluation on the dataset, the evaluates the final pipeline on the dataset.. """
    result, fit_time = timed(fit_func)()
    metrics = evaluate_func(result)
    return result, metrics, fit_time


def fit_evaluate_pipeline(pipeline, fit_func, evaluate_func) -> (Fedot, Dict[str, Any]):
    """ Runs Fedot evaluation on the dataset, the evaluates the final pipeline on the dataset.. """
    _, fit_time = timed(fit_func)()
    metrics = evaluate_func(pipeline)
    return pipeline, metrics, fit_time


def save_experiment_params(params_dict: Dict[str, Any], save_dir: Path):
    """ Save the hyperparameters of the experiment """
    params_file_path = save_dir.joinpath('parameters.json')
    with open(params_file_path, 'w') as params_file:
        json.dump(params_dict, params_file, indent=2)


def save_evaluation(save_dir: Path, dataset, pipeline, **kwargs):
    run_results: Dict[str, Any] = dict(dataset_id=dataset.id,
                                       dataset_name=dataset.name,
                                       model_obj=pipeline,
                                       model_str=pipeline.descriptive_id,
                                       task_type='classification',
                                       **kwargs)
    try:
        histories_dir = save_dir.joinpath('histories')
        models_dir = save_dir.joinpath('models')
        eval_results_path = save_dir.joinpath('evaluation_results.csv')

        histories_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)

        dataset_id = run_results['dataset_id']
        run_label = run_results['run_label']
        # define saving paths
        uid = str(uuid4())
        model_path = models_dir.joinpath(f'{dataset_id}_{run_label}_{uid}')
        history_path = histories_dir.joinpath(f'{dataset_id}_{run_label}_{uid}_history.json')
        # replace objects with export paths for csv
        run_results['model_path'] = str(model_path)
        run_results.pop('model_obj').save(model_path, create_subdir=False)
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


def run_fedot_attempt(train_data: TabularData, test_data: TabularData, timeout: float,
                      run_label: str, repetition: int, experiment_date: datetime, save_dir: Path,
                      fedot_evaluations_cache: CacheDict,
                      initial_assumption: Optional[Sequence[Pipeline]] = None, meta_learning_time_sec: float = 0.):
    fedot = Fedot(timeout=timeout, initial_assumption=initial_assumption, **FEDOT_PARAMS)
    fit_func = partial(fedot.fit, features=train_data.x, target=train_data.y)
    evaluate_func = partial(evaluate_pipeline, train_data=train_data, test_data=test_data)
    run_date = datetime.now()
    # cache_key = f'{run_label}_{train_data.id}_{timeout}_{repetition}'
    # with fedot_evaluations_cache as cache_dict:
    #     cached_run = cache_dict[cache_key]
    #     if cached_run:
    #         fedot = cached_run['fedot']
    #         pipeline = cached_run['pipeline']
    #         metrics = cached_run['metrics']
    #         fit_time = cached_run['fit_time']
    #     else:
    #         pipeline, metrics, fit_time = fit_evaluate_automl(fit_func=fit_func, evaluate_func=evaluate_func)
    #         cached_run = dict(
    #             fedot=fedot,
    #             pipeline=pipeline,
    #             metrics=metrics,
    #             fit_time=fit_time,
    #         )
    #         cache_dict[cache_key] = cached_run
    pipeline, metrics, fit_time = fit_evaluate_automl(fit_func=fit_func, evaluate_func=evaluate_func)
    eval_result = dict(
        dataset=train_data.dataset,
        run_label=run_label,
        pipeline=pipeline,
        meta_learning_time_sec=meta_learning_time_sec,
        automl_time_min=fit_time,
        automl_timeout_min=fedot.params.timeout,
        generations_count=fedot.history.generations_count,
        history_obj=fedot.history,
        run_data=run_date,
        experiment_date=experiment_date,
        save_dir=save_dir,
        **metrics
    )
    return eval_result


def run_pipeline(train_data: TabularData, test_data: TabularData, pipeline: Pipeline,
                 run_label: str, experiment_date: datetime, save_dir: Path):
    train_data_for_fedot = array_to_input_data(train_data.x, train_data.y)
    fit_func = partial(pipeline.fit, train_data_for_fedot)
    evaluate_func = partial(evaluate_pipeline, train_data=train_data, test_data=test_data)
    run_date = datetime.now()
    pipeline, metrics, fit_time = fit_evaluate_pipeline(pipeline=pipeline, fit_func=fit_func,
                                                        evaluate_func=evaluate_func)
    save_evaluation(dataset=train_data.dataset,
                    run_label=run_label,
                    pipeline=pipeline,
                    automl_time_min=0,
                    pipeline_fit_time=fit_time,
                    automl_timeout_min=0,
                    meta_learning_time_sec=0,
                    run_data=run_date,
                    experiment_date=experiment_date,
                    save_dir=save_dir,
                    **metrics)
    return pipeline


@loguru.logger.catch
def main():
    dataset_ids_test, dataset_ids_train, experiment_date, meta_learner_path, save_dir = setup_experiment()

    # fit_fedot_cached = CacheDict.decorate(fit_evaluate_automl, get_cache_dir() / 'fedot_runs.pkl', inner_key='dataset.id')
    dataset_splits = get_datasets_data_splits(dataset_ids_test + dataset_ids_train)
    datasets_eval_funcs = get_datasets_eval_funcs(dataset_ids_train, dataset_splits)

    algorithm = KNNSimilarityModelAdvice(
        N_BEST_DATASET_MODELS_TO_MEMORIZE,
        MF_EXTRACTOR_PARAMS,
        ASSESSOR_PARAMS,
        ADVISOR_PARAMS
    )

    # Experiment start
    knowledge_base = {dataset_id: [] for dataset_id in dataset_ids_train}
    fedot_evaluations_cache = CacheDict(get_cache_dir() / 'fedot_runs.pkl')
    description = 'FEDOT, train datasets'
    for dataset_id in (pbar := tqdm(dataset_ids_train, description)):
        pbar.set_description(description + f' ({dataset_id})')
        train_data, test_data = dataset_splits[dataset_id]['train'], dataset_splits[dataset_id]['test']
        run_label = 'FEDOT'
        evaluate_fedot_on_dataset(train_data, test_data, TRAIN_TIMEOUT, run_label, experiment_date, save_dir,
                                  fedot_evaluations_cache)
        # knowledge_base[dataset_id] = gain_knowledge_base_for_dataset(dataset_id, experiment_date,
        #                                                              fedot_evaluations_cache,
        #                                                              run_label, save_dir,
        #                                                              test_data, TRAIN_TIMEOUT, train_data)
        # knowledge_base[dataset_id] = [fedot.history for fedot in fedots]

    description = 'FEDOT, test datasets'
    for dataset_id in (pbar := tqdm(dataset_ids_test, description)):
        pbar.set_description(description + f' ({dataset_id})')
        train_data, test_data = dataset_splits[dataset_id]['train'], dataset_splits[dataset_id]['test']
        run_label = 'FEDOT'
        evaluate_fedot_on_dataset(train_data, test_data, TEST_TIMEOUT, run_label, experiment_date, save_dir,
                                  fedot_evaluations_cache)

    ###############################
    kb_datasets_data = [OpenMLDataset(dataset).get_data() for dataset in knowledge_base.keys()]
    kb_histories = list(knowledge_base.values())
    ###############################

    # Meta-Learning
    algorithm.fit(kb_datasets_data, kb_histories, datasets_eval_funcs)
    with open(meta_learner_path, 'wb') as meta_learner_file:
        pickle.dump(algorithm, meta_learner_file)
    # Application
    description = 'MetaFEDOT, Test datasets'
    for dataset_id in (pbar := tqdm(dataset_ids_test, description)):
        pbar.set_description(description + f' ({dataset_id})')
        train_data, test_data = dataset_splits[dataset_id]['train'], dataset_splits[dataset_id]['test']
        # Run meta AutoML
        # 1
        initial_assumptions, meta_learning_time_sec = timed(algorithm.predict, resolution='sec')([train_data])
        initial_assumptions = initial_assumptions[0]
        assumption_pipelines = [model.predictor for model in initial_assumptions]
        # 2
        baseline_pipeline = PipelineBuilder().add_node(BASELINE_MODEL).build()
        run_label = 'MetaFEDOT'
        evaluate_fedot_on_dataset(train_data, test_data, TEST_TIMEOUT, run_label, experiment_date, save_dir,
                                  fedot_evaluations_cache, assumption_pipelines, meta_learning_time_sec)
        # Fit & evaluate simple baseline
        run_label = 'simple baseline'
        try:
            run_pipeline(train_data, test_data, baseline_pipeline, run_label, experiment_date, save_dir)
        except Exception as e:
            logging.exception(f'Test dataset "{dataset_id}", {run_label}')
            if __debug__:
                raise e
        # Fit & evaluate initial assumptions
        for i, assumption in enumerate(initial_assumptions):
            try:
                pipeline = assumption.predictor
                run_label = f'MetaFEDOT - initial assumption {i}'
                run_pipeline(train_data, test_data, pipeline, run_label, experiment_date, save_dir)
            except Exception as e:
                logging.exception(f'Test dataset "{dataset_id}", {run_label}')
                if __debug__:
                    raise e


def get_datasets_eval_funcs(dataset_ids_train, dataset_splits):
    dataset_eval_funcs = []
    for dataset_id in dataset_ids_train:
        split = dataset_splits[dataset_id]
        train_data, test_data = split['train'], split['test']
        model_eval_func = partial(evaluate_pipeline, train_data=train_data, test_data=test_data, mode='fitness')
        dataset_eval_funcs.append(model_eval_func)
    return dataset_eval_funcs


def get_datasets_data_splits(dataset_ids):
    dataset_splits = {}
    for dataset_id in dataset_ids:
        dataset = OpenMLDataset(dataset_id)
        dataset_data = dataset.get_data()
        idx_train, idx_test = train_test_split(range(len(dataset_data.y)),
                                               test_size=DATA_TEST_SIZE,
                                               stratify=dataset_data.y,
                                               shuffle=True,
                                               random_state=DATA_SPLIT_SEED)
        train_data, test_data = dataset_data[idx_train], dataset_data[idx_test]
        dataset_splits[dataset_id] = dict(train=train_data, test=test_data)
    return dataset_splits


def setup_experiment():
    # Preparation
    experiment_date, experiment_date_iso, experiment_date_for_path = get_current_formatted_date()
    save_dir = get_save_dir(experiment_date_for_path)
    setup_logging(save_dir)
    if TMPDIR:
        os.environ.putenv('TMPDIR', TMPDIR)
    meta_learner_path = save_dir.joinpath('meta_learner.pkl')
    dataset_ids = get_dataset_ids()
    dataset_ids_train, dataset_ids_test = split_datasets(dataset_ids, N_DATASETS, UPDATE_TRAIN_TEST_DATASETS_SPLIT)
    dataset_ids = dataset_ids_train + dataset_ids_test
    experiment_params_dict = dict(
        experiment_start_date_iso=experiment_date_iso,
        input_config=config,
        dataset_ids=dataset_ids,
        dataset_ids_train=dataset_ids_train,
        dataset_ids_test=dataset_ids_test,
        baseline_pipeline=BASELINE_MODEL,
    )
    save_experiment_params(experiment_params_dict, save_dir)
    return dataset_ids_test, dataset_ids_train, experiment_date, meta_learner_path, save_dir


def evaluate_fedot_on_dataset(train_data: TabularData, test_data: TabularData, timeout: float,
                              run_label: str, experiment_date: datetime, save_dir: Path,
                              fedot_evaluations_cache: CacheDict,
                              initial_assumption: Optional[Sequence[Pipeline]] = None,
                              meta_learning_time_sec: float = 0.):
    dataset = train_data.dataset
    eval_results = []
    for repetition in range(N_AUTOML_REPETITIONS):
        try:
            eval_result, time_delta = timed(
                run_fedot_attempt(train_data, test_data, timeout, run_label, repetition, experiment_date, save_dir,
                                  fedot_evaluations_cache))
            # TODO:
            #   x Start FEDOT `N_BEST_DATASET_MODELS_TO_MEMORIZE` times, but not in one run

            # TODO: Условие на прерывание
            eval_results.append(eval_result)
        except Exception as e:
            logging.exception(f'Dataset "{dataset.id}"')
            if __debug__:
                raise e

    for eval_result in eval_results:
        save_evaluation(**eval_result)

    return eval_results


def gain_knowledge_base_for_dataset(train_data: TabularData, test_data: TabularData, timeout: float,
                                    run_label: str, experiment_date: datetime, save_dir: Path,
                                    fedot_evaluations_cache: CacheDict,
                                    initial_assumption: Optional[Sequence[Pipeline]] = None,
                                    meta_learning_time_sec: float = 0.):
    eval_results = evaluate_fedot_on_dataset(train_data, test_data, timeout,
                                             run_label, experiment_date, save_dir,
                                             fedot_evaluations_cache,
                                             initial_assumption,
                                             meta_learning_time_sec)
    histories = reduce([OptHistory.load, ], [res['history_path'] for res in eval_results])
    return histories


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception('Exception at main().')
        raise e
