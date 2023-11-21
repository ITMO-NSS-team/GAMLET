from __future__ import annotations

import functools
import json
import logging
import os
import pickle
import shutil
import timeit
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import openml
import pandas as pd
import yaml
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.quality_metrics_repository import MetricsRepository, QualityMetricsEnum
from fedot.core.validation.split import tabular_cv_generator
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from pecapiku import CacheDict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from meta_automl.approaches import MetaLearningApproach
from meta_automl.data_preparation.dataset import DatasetBase, DatasetData, DatasetIDType, OpenMLDataset
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
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

        datasets_loader = OpenMLDatasetsLoader()
        mf_extractor = PymfeExtractor(extractor_params=mf_extractor_params,
                                      datasets_loader=datasets_loader)
        datasets_similarity_assessor = KNeighborsSimilarityAssessor(**assessor_params)

        models_loader = FedotHistoryLoader()
        model_advisor = DiverseModelAdvisor(**advisor_params)

        self.components = self.Components(
            datasets_loader=datasets_loader,
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
        datasets: List[DatasetBase] = None
        dataset_ids: List[DatasetIDType] = None
        best_models: List[List[EvaluatedModel]] = None

    @dataclass
    class Components:
        datasets_loader: OpenMLDatasetsLoader
        models_loader: FedotHistoryLoader
        mf_extractor: PymfeExtractor
        mf_scaler: Any
        datasets_similarity_assessor: KNeighborsSimilarityAssessor
        model_advisor: DiverseModelAdvisor

    def load_models(self, dataset_ids: Sequence[DatasetIDType],
                    histories: Sequence[Sequence[OptHistory]]) -> List[List[EvaluatedModel]]:
        return self.components.models_loader.load(
            dataset_ids, histories,
            self.parameters.n_best_dataset_models_to_memorize
        )

    def extract_train_meta_features(self, dataset_ids: List[DatasetIDType]) -> pd.DataFrame:
        meta_features_train = self.components.mf_extractor.extract(
            dataset_ids, fill_input_nans=True)
        meta_features_train = meta_features_train.fillna(0)
        meta_features_train = pd.DataFrame(
            self.components.mf_scaler.fit_transform(meta_features_train),
            columns=meta_features_train.columns
        )
        return meta_features_train

    def fit_datasets_similarity_assessor(self, meta_features: pd.DataFrame, dataset_ids: List[DatasetIDType]
                                         ) -> KNeighborsSimilarityAssessor:
        return self.components.datasets_similarity_assessor.fit(meta_features, dataset_ids)

    def fit_model_advisor(self, dataset_ids: List[DatasetIDType], best_models: Sequence[Sequence[EvaluatedModel]]
                          ) -> DiverseModelAdvisor:
        return self.components.model_advisor.fit(dataset_ids, best_models)

    def fit(self, dataset_ids: Sequence[DatasetIDType], histories: Sequence[Sequence[OptHistory]]):
        d = self.data
        d.dataset_ids = list(dataset_ids)
        d.meta_features = self.extract_train_meta_features(d.dataset_ids)
        self.fit_datasets_similarity_assessor(d.meta_features, d.dataset_ids)
        d.best_models = self.load_models(dataset_ids, histories)
        self.fit_model_advisor(d.dataset_ids, d.best_models)
        return self

    def predict(self, datasets_ids: Sequence[DatasetIDType]) -> List[List[EvaluatedModel]]:
        extraction_params = dict(
            fill_input_nans=True, use_cached=False, update_cached=True
        )
        mf_extractor = self.components.mf_extractor
        mf_scaler = self.components.mf_scaler
        assessor = self.components.datasets_similarity_assessor
        advisor = self.components.model_advisor
        meta_features = mf_extractor.extract(datasets_ids, **extraction_params).fillna(0)
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


def get_current_formatted_date() -> (datetime, str, str):
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
                      input_data: InputData,
                      metrics: Sequence[QualityMetricsEnum] = COLLECT_METRICS_ENUM,
                      metric_names: Sequence[str] = COLLECT_METRICS) -> Dict[str, float]:
    """Gets quality metrics for the fitted pipeline.
    The function is based on `Fedot.get_metrics()`

    Returns:
        the values of quality metrics
    """
    data_producer = functools.partial(tabular_cv_generator, input_data, 10, StratifiedKFold)

    objective = MetricsObjective(metrics)
    obj_eval = PipelineObjectiveEvaluate(objective=objective,
                                         data_producer=data_producer,
                                         eval_n_jobs=-1)

    metric_values = obj_eval.evaluate(pipeline).values
    metric_values = {metric_name: round(value, 3) for (metric_name, value) in zip(metric_names, metric_values)}

    return metric_values


def transform_data_for_fedot(data: DatasetData) -> (np.array, np.array):
    x = data.x.to_numpy()
    y = data.y.to_numpy()
    return x, y


def fit_fedot(dataset: OpenMLDataset, timeout: float, run_label: str, initial_assumption=None) \
        -> (Fedot, Dict[str, Any]):
    """ Runs Fedot evaluation on the dataset, the evaluates the final pipeline on the dataset.
     Returns Fedot instance & properties of the run along with the evaluated metrics. """
    x, y = transform_data_for_fedot(dataset.get_data())

    time_start = timeit.default_timer()
    fedot = Fedot(timeout=timeout, initial_assumption=initial_assumption, logging_level=logging.DEBUG,
                  **COMMON_FEDOT_PARAMS)
    fedot.fit(x, y)
    automl_time = timeit.default_timer() - time_start

    metrics = evaluate_pipeline(fedot.current_pipeline, fedot.train_data)
    pipeline = fedot.current_pipeline
    run_results = get_result_data_row(dataset=dataset, run_label=run_label, pipeline=pipeline,
                                      automl_time_sec=automl_time, automl_timeout_min=fedot.params.timeout,
                                      history_obj=fedot.history, **metrics)
    return fedot, run_results


def get_result_data_row(dataset: OpenMLDataset, run_label: str, pipeline, history_obj=None, automl_time_sec=0.,
                        automl_timeout_min=0., **metrics) -> Dict[str, Any]:
    run_results = dict(dataset_id=dataset.id_,
                       dataset_name=dataset.name,
                       run_label=run_label,
                       model_obj=pipeline,
                       model_str=pipeline.descriptive_id,
                       history_obj=history_obj,
                       automl_time_sec=automl_time_sec,
                       automl_timeout_min=automl_timeout_min,
                       task_type='classification',
                       **metrics)
    return run_results


def save_experiment_params(params_dict: Dict[str, Any], save_dir: Path):
    """ Save the hyperparameters of the experiment """
    params_file_path = save_dir.joinpath('parameters.json')
    with open(params_file_path, 'w') as params_file:
        json.dump(params_dict, params_file, indent=2)


def save_evaluation(evaluation_properties: Dict[str, Any], run_date: datetime, experiment_date: datetime,
                    save_dir: Path):
    histories_dir = save_dir.joinpath('histories')
    models_dir = save_dir.joinpath('models')
    eval_results_path = save_dir.joinpath('evaluation_results.csv')

    histories_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    try:
        evaluation_properties['experiment_date'] = experiment_date
        evaluation_properties['run_date'] = run_date
        dataset_id = evaluation_properties['dataset_id']
        run_label = evaluation_properties['run_label']
        # define saving paths
        model_path = models_dir.joinpath(f'{dataset_id}_{run_label}')
        history_path = histories_dir.joinpath(f'{dataset_id}_{run_label}_history.json')
        # replace objects with export paths for csv
        evaluation_properties['model_path'] = str(model_path)
        evaluation_properties.pop('model_obj').save(model_path)
        evaluation_properties['history_path'] = str(history_path)
        history_obj = evaluation_properties.pop('history_obj')
        if history_obj is not None:
            history_obj.save(evaluation_properties['history_path'])

        df_evaluation_properties = pd.DataFrame([evaluation_properties])

        if eval_results_path.exists():
            df_results = pd.read_csv(eval_results_path)
            df_results = pd.concat([df_results, df_evaluation_properties])
        else:
            df_results = df_evaluation_properties
        df_results.to_csv(eval_results_path, index=False)

    except Exception:
        logging.exception(f'Saving results "{evaluation_properties}"')


def main():
    experiment_date, experiment_date_iso, experiment_date_for_path = get_current_formatted_date()
    save_dir = get_save_dir(experiment_date_for_path)
    setup_logging(save_dir)
    if TMPDIR:
        os.environ.putenv('TMPDIR', TMPDIR)
    progress_file_path = save_dir.joinpath('progress.txt')
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
    train_histories = {}
    fit_fedot_cached = CacheDict.decorate(fit_fedot, get_cache_dir() / 'fedot_runs.pkl', inner_key='dataset.id')
    with open(progress_file_path, 'a') as progress_file:
        description = 'FEDOT, all datasets'
        for dataset_id in (pbar := tqdm(dataset_ids, description, file=progress_file)):
            pbar.set_description(description + f' ({dataset_id})')
            try:
                timeout = TRAIN_TIMEOUT if dataset_id in dataset_ids_test else TEST_TIMEOUT
                dataset = algorithm.components.datasets_loader.load_single(dataset_id)
                run_date = datetime.now()
                fedot, run_results = fit_fedot_cached(dataset=dataset, timeout=timeout, run_label='FEDOT')
                save_evaluation(run_results, run_date, experiment_date, save_dir)
                # TODO:
                #   x Start FEDOT `N_BEST_DATASET_MODELS_TO_MEMORIZE` times, but not in one run
                if dataset_id not in dataset_ids_test:
                    history = fedot.history
                train_histories[dataset_id] = [history]
            except Exception:
                logging.exception(f'Train dataset "{dataset_id}"')

    # Learning
    algorithm.fit(list(train_histories.keys()), list(train_histories.values()))
    with open(meta_learner_path, 'wb') as meta_learner_file:
        pickle.dump(algorithm, meta_learner_file)

    fit_metafedot_cached = CacheDict.decorate(fit_fedot, fit_fedot, get_cache_dir() / 'metafedot_runs.pkl',
                                              inner_key='dataset.id')
    with open(progress_file_path, 'a') as progress_file:
        description = 'MetaFEDOT, Test datasets'
        for dataset_id in (pbar := tqdm(dataset_ids_test, description, file=progress_file)):
            pbar.set_description(description + f' ({dataset_id})')
            try:
                # Run meta AutoML
                # 1
                time_start = timeit.default_timer()
                initial_assumptions = algorithm.predict([dataset_id])[0]
                meta_learning_time_sec = timeit.default_timer() - time_start

                assumption_pipelines = [model.predictor for model in initial_assumptions]
                # 2
                dataset = algorithm.components.datasets_loader.load_single(dataset_id)
                run_date = datetime.now()
                fedot_meta, fedot_meta_results = fit_metafedot_cached(dataset=dataset,
                                                                      timeout=TEST_TIMEOUT,
                                                                      run_label='MetaFEDOT',
                                                                      initial_assumption=assumption_pipelines)
                fedot_meta_results['meta_learning_time_sec'] = meta_learning_time_sec
                save_evaluation(fedot_meta_results, run_date, experiment_date, save_dir)

                # Fit & evaluate simple baseline
                baseline_pipeline = PipelineBuilder().add_node(BASELINE_MODEL).build()
                run_date = datetime.now()
                baseline_metrics = evaluate_pipeline(baseline_pipeline, fedot_meta.train_data)
                baseline_res = get_result_data_row(dataset=dataset, run_label=f'simple baseline {BASELINE_MODEL}',
                                                   pipeline=baseline_pipeline,
                                                   **baseline_metrics)
                save_evaluation(baseline_res, run_date, experiment_date, save_dir)

                # Fit & evaluate initial assumptions
                for i, assumption in enumerate(initial_assumptions):
                    pipeline = assumption.predictor
                    run_date = datetime.now()
                    assumption_metrics = evaluate_pipeline(pipeline, fedot_meta.train_data)
                    assumption_res = get_result_data_row(dataset=dataset,
                                                         run_label=f'MetaFEDOT - initial assumption {i}',
                                                         pipeline=pipeline, **assumption_metrics)
                    save_evaluation(assumption_res, run_date, experiment_date, save_dir)
            except Exception:
                logging.exception(f'Test dataset "{dataset_id}"')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception('Main level caught an error.')
        raise
