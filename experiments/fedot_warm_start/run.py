import functools
import timeit
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Dict

import numpy as np
import openml
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.adapters import PipelineAdapter
from sklearn.model_selection import train_test_split
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
TEST_SIZE = 0.33
# Evaluation timeouts
TRAIN_TIMEOUT = 15
TEST_TIMEOUT = 10
# Models & datasets
N_BEST_DATASET_MODELS_TO_MEMORIZE = 10
N_CLOSEST_DATASETS_TO_PROPOSE = 5
MINIMAL_DISTANCE_BETWEEN_ADVISED_MODELS = 1
N_BEST_MODELS_TO_ADVISE = 5


COMMON_FEDOT_PARAMS = dict(
    problem='classification',
    with_tuning=False,
    logging_level=50,
    n_jobs=-1,
    seed=SEED,
)


def prepare_data() -> Dict[str, DatasetCache]:
    """Returns dictionary with dataset names and cached datasets downloaded from OpenML."""

    dataset_ids = openml.study.get_suite(99).data
    if N_DATASETS is not None:
        dataset_ids = pd.Series(dataset_ids)
        dataset_ids = dataset_ids.sample(n=N_DATASETS, random_state=SEED)
    dataset_ids = list(dataset_ids)
    return {cache.name: cache for cache in OpenMLDatasetsLoader().load(dataset_ids)}


def timeit_decorator(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        start_time = timeit.default_timer()
        res = function(*args, **kwargs)
        time = timeit.default_timer() - start_time
        return res, time

    return wrapped


def transform_data_for_fedot(data: Dataset) -> (np.array, np.array):
    x = data.x
    y = data.y
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return x, y


def main():
    datasets_cache = prepare_data()
    datasets_train, datasets_test = train_test_split(list(datasets_cache.keys()),
                                                     test_size=TEST_SIZE, random_state=SEED)

    extractor = PymfeExtractor(extractor_params={'groups': 'general'})
    meta_features_train = extractor.extract(datasets_train, fill_input_nans=True)
    meta_features_train = meta_features_train.fillna(0)
    data_similarity_assessor = KNeighborsBasedSimilarityAssessor(
        n_neighbors=min(len(datasets_train), N_CLOSEST_DATASETS_TO_PROPOSE))
    data_similarity_assessor.fit(meta_features_train, datasets_train)

    results_pre = []
    best_models_per_dataset = {}
    for name in tqdm(datasets_train, 'Train datasets'):
        cache = datasets_cache[name]
        data = cache.from_cache()

        fedot = Fedot(timeout=TRAIN_TIMEOUT, **COMMON_FEDOT_PARAMS)
        x, y = transform_data_for_fedot(data)
        _, automl_time = timeit_decorator(fedot.fit)(x, y)
        results_pre.append({'dataset': name,
                            'model': fedot.current_pipeline.descriptive_id,
                            'automl_time': automl_time})
        # TODO:
        #   x Turn the tuned pipeline into a model (evaluate its fitness on the data)
        #   x Evaluate historical pipelines on the data instead of using fitness

        # Filter out unique individuals with the best fitness
        best_individuals = sorted(chain(*fedot.history.individuals),
                                  key=lambda ind: ind.fitness,
                                  reverse=True)
        best_individuals = list({ind.graph.descriptive_id: ind for ind in best_individuals}.values())
        # best_models = list(fedot.best_models) or []
        best_models = []
        for individual in best_individuals[:N_BEST_DATASET_MODELS_TO_MEMORIZE]:
            pipeline = PipelineAdapter().restore(individual.graph)
            model = Model(pipeline, individual.fitness, cache)
            best_models.append(model)
        best_models_per_dataset[name] = best_models

    model_advisor = DiverseFEDOTPipelineAdvisor(data_similarity_assessor, n_best_to_advise=N_BEST_MODELS_TO_ADVISE,
                                                minimal_distance=MINIMAL_DISTANCE_BETWEEN_ADVISED_MODELS)
    model_advisor.fit(best_models_per_dataset)

    results = []
    for name in tqdm(datasets_test, 'Test datasets'):
        cache = datasets_cache[name]
        data = cache.from_cache()
        x, y = transform_data_for_fedot(data)

        fedot_naive = Fedot(timeout=TEST_TIMEOUT, **COMMON_FEDOT_PARAMS)
        _, automl_time_naive = timeit_decorator(fedot_naive.fit)(x, y)
        fedot_naive.test_data = fedot_naive.train_data
        fedot_naive.prediction = fedot_naive.train_data

        time_start = timeit.default_timer()
        meta_features = extractor.extract([cache], fill_input_nans=True, use_cached=False, update_cached=True)
        meta_features = meta_features.fillna(0)
        initial_assumptions = model_advisor.predict(meta_features)[0]
        initial_assumptions = [model.predictor for model in initial_assumptions]
        fedot_meta = Fedot(timeout=TEST_TIMEOUT, initial_assumption=initial_assumptions, **COMMON_FEDOT_PARAMS)
        fedot_meta.fit(x, y)
        automl_time_meta = timeit.default_timer() - time_start
        fedot_meta.test_data = fedot_meta.train_data
        fedot_meta.prediction = fedot_meta.train_data

        metrics_naive = fedot_naive.get_metrics()
        metrics_naive = {f'{key}_naive': val for key, val in metrics_naive.items()}
        metrics_meta = fedot_meta.get_metrics()
        metrics_meta = {f'{key}_meta': val for key, val in metrics_meta.items()}

        results.append({
            'dataset': data.name,
            'model_naive': fedot_naive.current_pipeline.descriptive_id,
            'model_meta': fedot_meta.current_pipeline.descriptive_id,
            'history_naive': fedot_naive.history,
            'history_meta': fedot_meta.history,
            'automl_time_naive': automl_time_naive,
            'automl_time_meta': automl_time_meta,
            **metrics_naive, **metrics_meta
        })

    time_now = datetime.now().isoformat(timespec="minutes").replace(":", ".")
    save_dir = Path(f'run_{time_now}')
    save_dir.mkdir()
    history_dir = save_dir.joinpath('histories')
    history_dir.mkdir()
    for res in results:
        dataset = res['dataset']
        res.pop('history_naive').save(history_dir.joinpath(f'{dataset}_history_naive.json'))
        res.pop('history_meta').save(history_dir.joinpath(f'{dataset}_history_meta.json'))
    pd.DataFrame(results_pre).to_csv(save_dir.joinpath(f'results_pre_{time_now}.csv'))
    pd.DataFrame(results).to_csv(save_dir.joinpath(f'results_{time_now}.csv'))


if __name__ == "__main__":
    main()
