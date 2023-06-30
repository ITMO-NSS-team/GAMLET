import functools
import json
import logging
import timeit
from datetime import datetime
from itertools import chain
from typing import Dict, List, Tuple, Sequence

import numpy as np
import openml
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.quality_metrics_repository import QualityMetricsEnum, MetricsRepository
from fedot.core.validation.split import tabular_cv_generator
from golem.core.log import Log
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


from meta_automl.data_preparation.dataset import OpenMLDataset, DatasetData
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.datasets_train_test_split import openml_datasets_train_test_split
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import KNeighborsBasedSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseFEDOTPipelineAdvisor

# Meta-alg hyperparameters
SEED = 42
# Datasets sampling
N_DATASETS = 3
TEST_SIZE = 0.33
# Evaluation timeouts
TRAIN_TIMEOUT = 0.01
TEST_TIMEOUT = 0.01
# Models & datasets
N_BEST_DATASET_MODELS_TO_MEMORIZE = 10
N_CLOSEST_DATASETS_TO_PROPOSE = 5
MINIMAL_DISTANCE_BETWEEN_ADVISED_MODELS = 1
N_BEST_MODELS_TO_ADVISE = 5
# Meta-features
MF_EXTRACTOR_PARAMS = {'groups': 'general'}
COLLECT_METRICS = ['f1', 'roc_auc', 'accuracy', 'neg_log_loss', 'precision']
COLLECT_METRICS_ENUM = tuple(map(MetricsRepository.metric_by_id, COLLECT_METRICS))
COLLECT_METRICS[COLLECT_METRICS.index('neg_log_loss')] = 'logloss'

COMMON_FEDOT_PARAMS = dict(
    problem='classification',
    n_jobs=-1,
    seed=SEED,
    show_progress=False,
)

# Setup logging
time_now = datetime.now()
time_now_iso = time_now.isoformat(timespec="minutes")
time_now_for_path = time_now_iso.replace(":", ".")
save_dir = get_data_dir(). \
    joinpath('experiments').joinpath('fedot_warm_start').joinpath(f'run_{time_now_for_path}')
save_dir.mkdir(parents=True)
log_file = save_dir.joinpath('log.txt')
Log(log_file=log_file)
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    force=True,
)


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, OpenMLDataset]]:
    """Returns dictionary with dataset names and cached datasets downloaded from OpenML."""

    dataset_ids = openml.study.get_suite(99).data
    if N_DATASETS is not None:
        dataset_ids = pd.Series(dataset_ids)
        dataset_ids = dataset_ids.sample(n=N_DATASETS, random_state=SEED)

    df_split_datasets = openml_datasets_train_test_split(dataset_ids, seed=SEED)
    df_datasets_train = df_split_datasets[df_split_datasets['is_train'] == 1]
    df_datasets_test = df_split_datasets[df_split_datasets['is_train'] == 0]

    datasets = {dataset.id_: dataset for dataset in OpenMLDatasetsLoader().load(dataset_ids)}
    return df_datasets_train, df_datasets_test, datasets


def transform_data_for_fedot(data: DatasetData) -> (np.array, np.array):
    x = data.x
    y = data.y
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return x, y


def get_pipeline_metrics(pipeline: Pipeline,
                         input_data: InputData,
                         metrics: Sequence[QualityMetricsEnum] = COLLECT_METRICS_ENUM,
                         metric_names: Sequence[str] = COLLECT_METRICS) -> dict:
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


def prepare_extractor_and_assessor(datasets_train: List[str]):
    extractor = PymfeExtractor(extractor_params=MF_EXTRACTOR_PARAMS)
    meta_features_train = extractor.extract(datasets_train, fill_input_nans=True)
    meta_features_train = meta_features_train.fillna(0)
    data_similarity_assessor = KNeighborsBasedSimilarityAssessor(
        n_neighbors=min(len(datasets_train), N_CLOSEST_DATASETS_TO_PROPOSE))
    data_similarity_assessor.fit(meta_features_train, datasets_train)
    return data_similarity_assessor, extractor


def fit_fedot(dataset: OpenMLDataset, timeout: float, run_label: str, initial_assumption=None):
    x, y = transform_data_for_fedot(dataset.get_data(dataset_format='array'))

    time_start = timeit.default_timer()
    fedot = Fedot(timeout=timeout, initial_assumption=initial_assumption, **COMMON_FEDOT_PARAMS)
    fedot.fit(x, y)
    automl_time = timeit.default_timer() - time_start

    metrics = get_pipeline_metrics(fedot.current_pipeline, fedot.train_data)
    pipeline = fedot.current_pipeline
    run_results = get_result_data_row(dataset=dataset, run_label=run_label, pipeline=pipeline, automl_time_sec=automl_time,
                                      automl_timeout_min=fedot.params.timeout, history_obj=fedot.history, **metrics)
    return fedot, run_results


def get_result_data_row(dataset: OpenMLDataset, run_label: str, pipeline, history_obj=None, automl_time_sec=0.,
                        automl_timeout_min=0., **metrics):
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


def extract_best_history_models(dataset, history):
    best_individuals = sorted(chain(*history.individuals),
                              key=lambda ind: ind.fitness,
                              reverse=True)
    best_individuals = list({ind.graph.descriptive_id: ind for ind in best_individuals}.values())
    best_models = []
    for individual in best_individuals[:N_BEST_DATASET_MODELS_TO_MEMORIZE]:
        pipeline = PipelineAdapter().restore(individual.graph)
        model = Model(pipeline, individual.fitness, history.objective.metric_names[0], dataset)
        best_models.append(model)
    return best_models


def main():
    baseline_pipeline = PipelineBuilder().add_node('rf').build()

    df_datasets_train, df_datasets_test, datasets = prepare_data()

    dataset_ids_train = df_datasets_train.index.to_list()
    dataset_ids_test = df_datasets_test.index.to_list()

    evaluation_results = []
    best_models_per_dataset = {}
    progress_file = open(save_dir.joinpath('progress.txt'), 'a')
    for dataset_id in tqdm(datasets.keys(), 'FEDOT, all datasets', file=progress_file):
        try:
            dataset = datasets[dataset_id]
            timeout = TRAIN_TIMEOUT if dataset_id in dataset_ids_train else TEST_TIMEOUT
            fedot, run_results = fit_fedot(dataset=dataset, timeout=timeout, run_label='FEDOT')
            evaluation_results.append(run_results)
            # TODO:
            #   x Turn the tuned pipeline into a model (evaluate its fitness on the data)
            #   x Evaluate historical pipelines on the data instead of using fitness
            #   x Start FEDOT `N_BEST_DATASET_MODELS_TO_MEMORIZE` times, but not in one run

            # Filter out unique individuals with the best fitness
            history = fedot.history
            best_models = extract_best_history_models(dataset, history)
            best_models_per_dataset[dataset_id] = best_models
        except Exception:
            logging.exception(f'Train dataset "{dataset_id}"')

    data_similarity_assessor, extractor = prepare_extractor_and_assessor(dataset_ids_train)
    model_advisor = DiverseFEDOTPipelineAdvisor(data_similarity_assessor, n_best_to_advise=N_BEST_MODELS_TO_ADVISE,
                                                minimal_distance=MINIMAL_DISTANCE_BETWEEN_ADVISED_MODELS)
    model_advisor.fit(best_models_per_dataset)

    for dataset_id in tqdm(dataset_ids_test, 'MetaFEDOT, Test datasets', file=progress_file):
        try:
            dataset = datasets[dataset_id]

            # Run meta AutoML
            # 1
            time_start = timeit.default_timer()
            meta_features = extractor.extract([dataset], fill_input_nans=True, use_cached=False, update_cached=True)
            meta_features = meta_features.fillna(0)
            meta_learning_time_sec = timeit.default_timer() - time_start
            initial_assumptions = model_advisor.predict(meta_features)[0]
            assumption_pipelines = [model.predictor for model in initial_assumptions]
            # 2
            fedot_meta, fedot_meta_results = fit_fedot(dataset=dataset, timeout=TEST_TIMEOUT, run_label='MetaFEDOT',
                                                       initial_assumption=assumption_pipelines)
            fedot_meta_results['meta_learning_time_sec'] = meta_learning_time_sec
            evaluation_results.append(fedot_meta_results)

            # Fit & evaluate simple baseline
            baseline_metrics = get_pipeline_metrics(baseline_pipeline, fedot_meta.train_data)
            baseline_res = get_result_data_row(dataset=dataset, run_label='simple baseline', pipeline=baseline_pipeline,
                                               **baseline_metrics)
            evaluation_results.append(baseline_res)

            # Fit & evaluate initial assumptions
            for i, assumption in enumerate(initial_assumptions):
                pipeline = assumption.predictor
                assumption_metrics = get_pipeline_metrics(pipeline, fedot_meta.train_data)
                assumption_res = get_result_data_row(dataset=dataset, run_label=f'MetaFEDOT - initial assumption {i}',
                                                     pipeline=pipeline, **assumption_metrics)
                evaluation_results.append(assumption_res)
        except Exception:
            logging.exception(f'Test dataset "{dataset_id}"')
    progress_file.close()

    # Save the accumulated results
    history_dir = save_dir.joinpath('histories')
    history_dir.mkdir()
    models_dir = save_dir.joinpath('models')
    for res in evaluation_results:
        try:
            res['run_date'] = time_now
            dataset_id = res['dataset_id']
            run_label = res['run_label']
            # define saving paths
            model_path = models_dir.joinpath(f'{dataset_id}_{run_label}')
            history_path = history_dir.joinpath(f'{dataset_id}_{run_label}_history.json')
            # replace objects with export paths for csv
            res['model_path'] = str(model_path)
            res.pop('model_obj').save(res['model_path'])
            res['history_path'] = str(history_path)
            history_obj = res.pop('history_obj')
            if history_obj is not None:
                history_obj.save(res['history_path'])
        except Exception:
            logging.exception(f'Saving results "{res}"')

    pd.DataFrame(evaluation_results).to_csv(save_dir.joinpath(f'results_{time_now_for_path}.csv'))

    # save experiment hyperparameters
    params = {
        'run_date': time_now_iso,
        'seed': SEED,
        'n_datasets': N_DATASETS or len(datasets),
        'test_size': TEST_SIZE,
        'dataset_ids': list(datasets.keys()),
        'dataset_ids_train': dataset_ids_train,
        'dataset_ids_test': dataset_ids_test,
        'dataset_names_train': df_datasets_train['dataset_name'].to_list(),
        'dataset_names_test': df_datasets_test['dataset_name'].to_list(),
        'train_timeout': TRAIN_TIMEOUT,
        'test_timeout': TEST_TIMEOUT,
        'n_best_dataset_models_to_memorize': N_BEST_DATASET_MODELS_TO_MEMORIZE,
        'n_closest_datasets_to_propose': N_CLOSEST_DATASETS_TO_PROPOSE,
        'minimal_distance_between_advised_models': MINIMAL_DISTANCE_BETWEEN_ADVISED_MODELS,
        'n_best_models_to_advise': N_BEST_MODELS_TO_ADVISE,
        'common_fedot_params': COMMON_FEDOT_PARAMS,
        'baseline_pipeline': baseline_pipeline.descriptive_id,
    }
    with open(save_dir.joinpath('parameters.json'), 'w') as params_file:
        json.dump(params, params_file, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception('Main level caught an error.')
        raise
