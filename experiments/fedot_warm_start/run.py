import functools
import json
import logging
import timeit
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openml
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.validation.split import tabular_cv_generator
from golem.core.log import Log
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

from meta_automl.data_preparation.dataset import DatasetCache, Dataset
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import KNeighborsBasedSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseFEDOTPipelineAdvisor

# Meta-alg hyperparameters
SEED = 42
# Datasets sampling
N_DATASETS = None
TEST_SIZE = 0.2
# Evaluation timeouts
TRAIN_TIMEOUT = 15
TEST_TIMEOUT = 15
# Models & datasets
N_BEST_DATASET_MODELS_TO_MEMORIZE = 10
N_CLOSEST_DATASETS_TO_PROPOSE = 5
MINIMAL_DISTANCE_BETWEEN_ADVISED_MODELS = 1
N_BEST_MODELS_TO_ADVISE = 5
# Meta-features
MF_EXTRACTOR_PARAMS = {'groups': 'general'}

COMMON_FEDOT_PARAMS = dict(
    problem='classification',
    n_jobs=-1,
    seed=SEED,
    show_progress=False,
)

# Setup logging
time_now = datetime.now().isoformat(timespec="minutes")
time_now_for_path = time_now.replace(":", ".")
save_dir = Path(f'run_{time_now_for_path}')
save_dir.mkdir()
log_file = save_dir.joinpath('log.txt')
Log(log_file=log_file)
logging.basicConfig(filename=log_file,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    force=True,
                    )


def prepare_data() -> Tuple[List[int], Dict[str, DatasetCache]]:
    """Returns dictionary with dataset names and cached datasets downloaded from OpenML."""

    dataset_ids = openml.study.get_suite(99).data
    if N_DATASETS is not None:
        dataset_ids = pd.Series(dataset_ids)
        dataset_ids = dataset_ids.sample(n=N_DATASETS, random_state=SEED)
    dataset_ids = list(dataset_ids)
    return dataset_ids, {cache.name: cache for cache in OpenMLDatasetsLoader().load(dataset_ids)}


def transform_data_for_fedot(data: Dataset) -> (np.array, np.array):
    x = data.x
    y = data.y
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return x, y


def get_pipeline_metrics(pipeline,
                         input_data,
                         metrics_obj) -> dict:
    """Gets quality metrics for the fitted pipeline.
    The function is based on `Fedot.get_metrics()`

    Returns:
        the values of quality metrics
    """
    metrics = metrics_obj.metric_functions
    metric_names = metrics_obj.get_metric_names(metrics)

    data_producer = functools.partial(tabular_cv_generator, input_data, 10, StratifiedKFold)

    objective = MetricsObjective(metrics)
    obj_eval = PipelineObjectiveEvaluate(objective=objective,
                                         data_producer=data_producer,
                                         eval_n_jobs=-1)

    metrics = obj_eval.evaluate(pipeline).values
    metrics = {metric_name: round(metric, 3) for (metric_name, metric) in zip(metric_names, metrics)}

    return metrics


def prepare_extractor_and_assessor(datasets_train: List[str]):
    extractor = PymfeExtractor(extractor_params=MF_EXTRACTOR_PARAMS)
    meta_features_train = extractor.extract(datasets_train, fill_input_nans=True)
    meta_features_train = meta_features_train.fillna(0)
    data_similarity_assessor = KNeighborsBasedSimilarityAssessor(
        n_neighbors=min(len(datasets_train), N_CLOSEST_DATASETS_TO_PROPOSE))
    data_similarity_assessor.fit(meta_features_train, datasets_train)
    return data_similarity_assessor, extractor


def fit_fedot(data: Dataset, timeout: float, run_label: str, initial_assumption=None):
    x, y = transform_data_for_fedot(data)

    time_start = timeit.default_timer()
    fedot = Fedot(timeout=timeout, initial_assumption=initial_assumption, **COMMON_FEDOT_PARAMS)
    fedot.fit(x, y)
    automl_time = timeit.default_timer() - time_start

    metrics = get_pipeline_metrics(fedot.current_pipeline, fedot.train_data, fedot.metrics)
    pipeline = fedot.current_pipeline
    run_results = get_result_data_row(dataset=data, run_label=run_label, pipeline=pipeline, automl_time_sec=automl_time,
                                      automl_timeout_min=fedot.params.timeout, history_obj=fedot.history, **metrics)
    return fedot, run_results


def get_result_data_row(dataset, run_label: str, pipeline, history_obj=None, automl_time_sec=0., automl_timeout_min=0.,
                        **metrics):
    run_results = dict(dataset_id=dataset.id,
                       dataset_name=dataset.name,
                       run_label=run_label,
                       model_obj=pipeline,
                       model_str=pipeline.descriptive_id,
                       history_obj=history_obj,
                       automl_time_sec=automl_time_sec,
                       automl_timeout_min=automl_timeout_min,
                       **metrics)
    return run_results


def extract_best_history_models(dataset_cache, history):
    best_individuals = sorted(chain(*history.individuals),
                              key=lambda ind: ind.fitness,
                              reverse=True)
    best_individuals = list({ind.graph.descriptive_id: ind for ind in best_individuals}.values())
    best_models = []
    for individual in best_individuals[:N_BEST_DATASET_MODELS_TO_MEMORIZE]:
        pipeline = PipelineAdapter().restore(individual.graph)
        model = Model(pipeline, individual.fitness, dataset_cache)
        best_models.append(model)
    return best_models


def main():
    baseline_pipeline = PipelineBuilder().add_node('rf').build()

    dataset_ids, datasets_cache = prepare_data()

    datasets_train, datasets_test = \
        train_test_split(list(datasets_cache.keys()), test_size=TEST_SIZE, random_state=SEED)

    data_similarity_assessor, extractor = prepare_extractor_and_assessor(datasets_train)

    results = []
    best_models_per_dataset = {}
    progress_file = open(save_dir.joinpath('progress.txt'), 'a')
    for name in tqdm(datasets_train, 'Train datasets', file=progress_file):
        try:
            cache = datasets_cache[name]
            data = cache.from_cache()

            fedot, run_results = fit_fedot(data=data, timeout=TRAIN_TIMEOUT, run_label='FEDOT')
            results.append(run_results)
            # TODO:
            #   x Turn the tuned pipeline into a model (evaluate its fitness on the data)
            #   x Evaluate historical pipelines on the data instead of using fitness
            #   x Start FEDOT `N_BEST_DATASET_MODELS_TO_MEMORIZE` times, but not in one run

            # Filter out unique individuals with the best fitness
            history = fedot.history
            best_models = extract_best_history_models(cache, history)
            best_models_per_dataset[name] = best_models
        except Exception:
            logging.exception(f'Train dataset "{name}"')

    model_advisor = DiverseFEDOTPipelineAdvisor(data_similarity_assessor, n_best_to_advise=N_BEST_MODELS_TO_ADVISE,
                                                minimal_distance=MINIMAL_DISTANCE_BETWEEN_ADVISED_MODELS)
    model_advisor.fit(best_models_per_dataset)

    for name in tqdm(datasets_test, 'Test datasets', file=progress_file):
        try:
            cache = datasets_cache[name]
            data = cache.from_cache()

            # Run pure AutoML
            fedot_naive, fedot_naive_results = fit_fedot(data=data, timeout=TEST_TIMEOUT, run_label='FEDOT')
            results.append(fedot_naive_results)

            # Run meta AutoML
            # 1
            time_start = timeit.default_timer()
            meta_features = extractor.extract([cache], fill_input_nans=True, use_cached=False, update_cached=True)
            meta_features = meta_features.fillna(0)
            meta_learning_time = timeit.default_timer() - time_start
            initial_assumptions = model_advisor.predict(meta_features)[0]
            assumption_pipelines = [model.predictor for model in initial_assumptions]
            # 2
            fedot_meta, fedot_meta_results = fit_fedot(data=data, timeout=TEST_TIMEOUT, run_label='MetaFEDOT',
                                                       initial_assumption=assumption_pipelines)
            fedot_meta_results['meta_learning_time'] = meta_learning_time
            results.append(fedot_meta_results)

            # Fit & evaluate simple baseline
            baseline_metrics = get_pipeline_metrics(baseline_pipeline, fedot_meta.train_data, fedot_meta.metrics)
            baseline_res = get_result_data_row(dataset=data, run_label='simple baseline', pipeline=baseline_pipeline,
                                               **baseline_metrics)
            results.append(baseline_res)

            # Fit & evaluate initial assumptions
            for i, assumption in enumerate(initial_assumptions):
                pipeline = assumption.predictor
                assumption_metrics = get_pipeline_metrics(pipeline, fedot_meta.train_data, fedot_meta.metrics)
                assumption_res = get_result_data_row(dataset=data, run_label=f'MetaFEDOT - initial assumption {i}',
                                                     pipeline=pipeline, **assumption_metrics)
                results.append(assumption_res)
        except Exception:
            logging.exception(f'Test dataset "{name}"')

    # Save the accumulated results
    history_dir = save_dir.joinpath('histories')
    history_dir.mkdir()
    models_dir = save_dir.joinpath('models')
    for res in results:
        try:
            res['run_date'] = time_now
            dataset_name = res['dataset_name']
            run_label = res['run_label']
            # define saving paths
            model_path = models_dir.joinpath(f'{dataset_name}_{run_label}')
            history_path = history_dir.joinpath(f'{dataset_name}_{run_label}_history.json')
            # replace objects with export paths for csv
            res['model_path'] = str(model_path)
            res.pop('model_obj').save(res['model_path'])
            res['history_path'] = str(history_path)
            history_obj = res.pop('history_obj')
            if history_obj is not None:
                history_obj.save(res['history_path'])
        except Exception:
            logging.exception(f'Saving results "{res}"')

    pd.DataFrame(results).to_csv(save_dir.joinpath(f'results_{time_now_for_path}.csv'))

    # save experiment hyperparameters
    params = {
        'run_date': time_now,
        'seed': SEED,
        'n_datasets': N_DATASETS or len(dataset_ids),
        'test_size': TEST_SIZE,
        'dataset_ids': dataset_ids,
        'dataset_names': list(datasets_cache.keys()),
        'dataset_names_train': datasets_train,
        'dataset_names_test': datasets_test,
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
    except Exception:
        logging.exception(f'Main level cached the error')
