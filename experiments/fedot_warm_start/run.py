from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import timeit
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import openml
import pandas as pd
import yaml
from fedot.api.main import Fedot
from fedot.core.data.data import array_to_input_data
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.quality_metrics_repository import MetricsRepository, QualityMetricsEnum
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from pecapiku import CacheDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from meta_automl.approaches import MetaLearningApproach
from meta_automl.data_preparation.dataset import DatasetIDType, OpenMLDataset, TabularData
from meta_automl.data_preparation.datasets_train_test_split import openml_datasets_train_test_split
from meta_automl.data_preparation.evaluated_model import EvaluatedModel
from meta_automl.data_preparation.file_system import get_cache_dir
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.models_loaders import FedotHistoryLoader
from meta_automl.meta_algorithm.dataset_similarity_assessors import KNeighborsSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseModelAdvisor

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
# Optional values
TMPDIR = config.get('tmpdir')
SAVE_DIR_PREFIX = config.get('save_dir_prefix')

UPDATE_TRAIN_TEST_DATASETS_SPLIT = config.get('update_train_test_datasets_split')

# Postprocess constants
COLLECT_METRICS_ENUM = tuple(map(MetricsRepository.metric_by_id, COLLECT_METRICS))
COLLECT_METRICS[COLLECT_METRICS.index('neg_log_loss')] = 'logloss'


class KNNSimilarityAdvice(MetaLearningApproach):
    def __init__(self, n_best_dataset_models_to_memorize: int,
                 mf_extractor_params: dict, assessor_params: dict, advisor_params: dict):
        self.parameters = self.Parameters(
            n_best_dataset_models_to_memorize=n_best_dataset_models_to_memorize,
            mf_extractor_params=mf_extractor_params,
            assessor_params=assessor_params,
            advisor_params=advisor_params,
        )
        self.data = self.Data()

        mf_extractor = PymfeExtractor(extractor_params=mf_extractor_params)
        datasets_similarity_assessor = KNeighborsSimilarityAssessor(**assessor_params)

        models_loader = FedotHistoryLoader()
        model_advisor = DiverseModelAdvisor(**advisor_params)

        self.components = self.Components(
            models_loader=models_loader,
            mf_extractor=mf_extractor,
            mf_scaler=MinMaxScaler(),
            datasets_similarity_assessor=datasets_similarity_assessor,
            model_advisor=model_advisor,
        )

    @dataclass
    class Parameters:
        n_best_dataset_models_to_memorize: int
        mf_extractor_params: dict = field(default_factory=dict)
        assessor_params: dict = field(default_factory=dict)
        advisor_params: dict = field(default_factory=dict)

    @dataclass
    class Data:
        meta_features: pd.DataFrame = None
        datasets: List[OpenMLDataset] = None
        datasets_data: List[OpenMLDataset] = None
        dataset_ids: List[DatasetIDType] = None
        best_models: List[List[EvaluatedModel]] = None

    @dataclass
    class Components:
        models_loader: FedotHistoryLoader
        mf_extractor: PymfeExtractor
        mf_scaler: Any
        datasets_similarity_assessor: KNeighborsSimilarityAssessor
        model_advisor: DiverseModelAdvisor

    def fit(self,
            datasets_data: Sequence[TabularData],
            histories: Sequence[Sequence[OptHistory]],
            evaluate_model_func: Optional[Sequence[Callable]] = None):
        data = self.data
        params = self.parameters

        data.datasets_data = list(datasets_data)
        data.datasets = [d.dataset for d in datasets_data]
        data.dataset_ids = [d.id for d in datasets_data]

        data.meta_features = self.extract_train_meta_features(data.datasets_data)
        self.fit_datasets_similarity_assessor(data.meta_features, data.dataset_ids)

        data.best_models = self.load_models(data.datasets, histories, params.n_best_dataset_models_to_memorize,
                                            evaluate_model_func)
        self.fit_model_advisor(data.dataset_ids, data.best_models)

        return self

    def load_models(
            self, datasets: Sequence[OpenMLDataset],
            histories: Sequence[Sequence[OptHistory]],
            n_best_dataset_models_to_load: int,
            evaluate_model_func: Optional[Sequence[Callable]] = None) -> List[List[EvaluatedModel]]:
        return self.components.models_loader.load(datasets, histories, n_best_dataset_models_to_load,
                                                  evaluate_model_func)

    def extract_train_meta_features(self, datasets_data: List[TabularData]) -> pd.DataFrame:
        components = self.components

        meta_features = components.mf_extractor.extract(
            datasets_data, fill_input_nans=True)

        meta_features.fillna(0, inplace=True)

        meta_features = pd.DataFrame(components.mf_scaler.fit_transform(meta_features), columns=meta_features.columns)

        return meta_features

    def fit_datasets_similarity_assessor(self, meta_features: pd.DataFrame, dataset_ids: List[DatasetIDType]
                                         ) -> KNeighborsSimilarityAssessor:
        return self.components.datasets_similarity_assessor.fit(meta_features, dataset_ids)

    def fit_model_advisor(self, dataset_ids: List[DatasetIDType], best_models: Sequence[Sequence[EvaluatedModel]]
                          ) -> DiverseModelAdvisor:
        return self.components.model_advisor.fit(dataset_ids, best_models)

    def predict(self, datasets_data: Sequence[TabularData]) -> List[List[EvaluatedModel]]:
        mf_extractor = self.components.mf_extractor
        mf_scaler = self.components.mf_scaler
        assessor = self.components.datasets_similarity_assessor
        advisor = self.components.model_advisor

        meta_features = mf_extractor.extract(datasets_data, fill_input_nans=True)

        meta_features.fillna(0, inplace=True)

        meta_features = pd.DataFrame(mf_scaler.transform(meta_features), columns=meta_features.columns)

        similar_dataset_ids = assessor.predict(meta_features)

        return advisor.predict(similar_dataset_ids)


def setup_logging(save_dir: Path):
    """ Creates "log.txt" at the "save_dir" and redirects all logging output to it. """
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


def run_fedot(train_data: TabularData, test_data: TabularData, timeout: float,
              run_label: str, experiment_date: datetime, save_dir: Path, fedot_evaluations_cache: CacheDict,
              initial_assumption: Optional[Sequence[Pipeline]] = None, meta_learning_time_sec: float = 0.):
    fedot = Fedot(timeout=timeout, initial_assumption=initial_assumption, **FEDOT_PARAMS)
    fit_func = partial(fedot.fit, features=train_data.x, target=train_data.y)
    evaluate_func = partial(evaluate_pipeline, train_data=train_data, test_data=test_data)
    run_date = datetime.now()
    cache_key = f'{run_label}_{train_data.id}'
    with fedot_evaluations_cache as cache_dict:
        cached_run = cache_dict[cache_key]
        if cached_run:
            fedot = cached_run['fedot']
            pipeline = cached_run['pipeline']
            metrics = cached_run['metrics']
            fit_time = cached_run['fit_time']
        else:
            pipeline, metrics, fit_time = fit_evaluate_automl(fit_func=fit_func, evaluate_func=evaluate_func)
            cached_run = dict(
                fedot=fedot,
                pipeline=pipeline,
                metrics=metrics,
                fit_time=fit_time,
            )
            cache_dict[cache_key] = cached_run
    save_evaluation(dataset=train_data.dataset,
                    run_label=run_label,
                    pipeline=pipeline,
                    meta_learning_time_sec=meta_learning_time_sec,
                    automl_time_min=fit_time,
                    automl_timeout_min=fedot.params.timeout,
                    history_obj=fedot.history,
                    run_data=run_date,
                    experiment_date=experiment_date,
                    save_dir=save_dir,
                    **metrics)
    return fedot


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


def main():
    experiment_date, experiment_date_iso, experiment_date_for_path = get_current_formatted_date()
    save_dir = get_save_dir(experiment_date_for_path)
    setup_logging(save_dir)
    if TMPDIR:
        os.environ.putenv('TMPDIR', TMPDIR)
    meta_learner_path = save_dir.joinpath('meta_learner.pkl')

    dataset_ids = get_dataset_ids()
    dataset_ids_train, dataset_ids_test = split_datasets(dataset_ids, N_DATASETS, UPDATE_TRAIN_TEST_DATASETS_SPLIT)
    dataset_ids = dataset_ids_train + dataset_ids_test

    algorithm = KNNSimilarityAdvice(
        N_BEST_DATASET_MODELS_TO_MEMORIZE,
        MF_EXTRACTOR_PARAMS,
        ASSESSOR_PARAMS,
        ADVISOR_PARAMS
    )

    experiment_params_dict = dict(
        experiment_start_date_iso=experiment_date_iso,
        input_config=config,
        dataset_ids=dataset_ids,
        dataset_ids_train=dataset_ids_train,
        dataset_ids_test=dataset_ids_test,
        baseline_pipeline=BASELINE_MODEL,
    )
    save_experiment_params(experiment_params_dict, save_dir)
    # Gathering knowledge base
    # fit_fedot_cached = CacheDict.decorate(fit_evaluate_automl, get_cache_dir() / 'fedot_runs.pkl', inner_key='dataset.id')
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

    knowledge_base = {}
    fedot_evaluations_cache = CacheDict(get_cache_dir() / 'fedot_runs.pkl', access='rew')
    description = 'FEDOT, all datasets'
    for dataset_id in (pbar := tqdm(dataset_ids, description)):
        pbar.set_description(description + f' ({dataset_id})')
        try:
            timeout = TRAIN_TIMEOUT if dataset_id in dataset_ids_test else TEST_TIMEOUT
            train_data, test_data = dataset_splits[dataset_id]['train'], dataset_splits[dataset_id]['test']
            run_label = 'FEDOT'
            fedot = run_fedot(train_data, test_data, timeout, run_label, experiment_date, save_dir,
                              fedot_evaluations_cache)
            # TODO:
            #   x Start FEDOT `N_BEST_DATASET_MODELS_TO_MEMORIZE` times, but not in one run
            if dataset_id not in dataset_ids_test:
                if fedot.history:
                    knowledge_base[dataset_id] = [fedot.history]
        except Exception as e:
            logging.exception(f'Train dataset "{dataset_id}"')
            if __debug__:
                raise e
    knowledge_base_data = [OpenMLDataset(dataset).get_data() for dataset in knowledge_base.keys()]
    knowledge_base_histories = list(knowledge_base.values())
    # Learning
    dataset_eval_funcs = []
    for dataset_id in dataset_ids_train:
        split = dataset_splits[dataset_id]
        train_data, test_data = split['train'], split['test']
        model_eval_func = partial(evaluate_pipeline, train_data=train_data, test_data=test_data, mode='fitness')
        dataset_eval_funcs.append(model_eval_func)
    algorithm.fit(knowledge_base_data, knowledge_base_histories, dataset_eval_funcs)
    with open(meta_learner_path, 'wb') as meta_learner_file:
        pickle.dump(algorithm, meta_learner_file)

    description = 'MetaFEDOT, Test datasets'
    for dataset_id in (pbar := tqdm(dataset_ids_test, description)):
        pbar.set_description(description + f' ({dataset_id})')
        try:
            train_data, test_data = dataset_splits[dataset_id]['train'], dataset_splits[dataset_id]['test']
            # Run meta AutoML
            # 1
            initial_assumptions, meta_learning_time_sec = timed(algorithm.predict, resolution='sec')([train_data])
            initial_assumptions = initial_assumptions[0]
            assumption_pipelines = [model.predictor for model in initial_assumptions]
            # 2
            timeout = TRAIN_TIMEOUT if dataset_id in dataset_ids_test else TEST_TIMEOUT
            run_label = 'MetaFEDOT'
            run_fedot(train_data, test_data, timeout, run_label, experiment_date, save_dir,
                      fedot_evaluations_cache, initial_assumption=assumption_pipelines,
                      meta_learning_time_sec=meta_learning_time_sec)
            # Fit & evaluate simple baseline
            baseline_pipeline = PipelineBuilder().add_node(BASELINE_MODEL).build()
            run_label = 'simple baseline'
            run_pipeline(train_data, test_data, baseline_pipeline, run_label, experiment_date, save_dir)

            # Fit & evaluate initial assumptions
            for i, assumption in enumerate(initial_assumptions):
                pipeline = assumption.predictor
                run_label = f'MetaFEDOT - initial assumption {i}'
                run_pipeline(train_data, test_data, pipeline, run_label, experiment_date, save_dir)
        except Exception as e:
            logging.exception(f'Test dataset "{dataset_id}"')
            if __debug__:
                raise e


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception('Exception at main().')
        raise e
