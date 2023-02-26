import functools
import timeit

import openml
import pandas as pd
from fedot.api.main import Fedot
from sklearn.model_selection import train_test_split

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor

SEED = 42


def prepare_data():
    dataset_ids = pd.Series(openml.study.get_suite(99).data)
    dataset_ids = dataset_ids.sample(n=15, random_state=SEED)
    dataset_ids = list(dataset_ids)
    return OpenMLDatasetsLoader().load(dataset_ids)


def timeit_decorator(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        start_time = timeit.default_timer()
        res = function(*args, **kwargs)
        time = timeit.default_timer() - start_time
        return res, time

    return wrapped


def main():
    datasets_cache = prepare_data()
    datasets_train, datasets_test = train_test_split(datasets_cache, test_size=0.33, random_state=SEED)

    # TODO:
    #  - Extract meta-features for train datasets
    #  - Fit 'DatasetsSimilarityAssessor'

    results_pre = []
    for cache in datasets_train:
        data = cache.from_cache()
        fedot = Fedot('classification', timeout=15, n_jobs=-1, seed=SEED)
        _, automl_time = timeit_decorator(fedot.fit)(data.x, data.y)
        results_pre.append({'dataset': data.name, 'model': fedot, 'automl_time': automl_time})

    # TODO:
    #  - Prepare 'ModelAdvisor'

    results = []
    for cache in datasets_test:
        data = cache.from_cache()
        fedot_naive = Fedot('classification', timeout=5, n_jobs=-1, seed=SEED)
        _, automl_time_naive = timeit_decorator(fedot_naive.fit)(data.x, data.y)

        time_start = timeit.default_timer()
        # TODO:
        #  - Extract meta-features for current test dataset
        #  - Get suitable assumptions from 'ModelAdvisor'
        initial_assumption = ...
        fedot_meta = Fedot('classification', timeout=5, n_jobs=-1, seed=SEED, initial_assumption=initial_assumption)
        automl_time_meta = timeit.default_timer() - time_start

        metrics_naive = fedot_naive.get_metrics()
        metrics_naive = {f'{key}_naive': val for key, val in metrics_naive.items()}
        metrics_meta = fedot_meta.get_metrics()
        metrics_meta = {f'{key}_meta': val for key, val in metrics_meta.items()}

        results.append({
            'dataset': data.name,
            'model_naive': fedot_naive,
            'model_meta': fedot_meta,
            'automl_time_naive': automl_time_naive,
            'automl_time_meta': automl_time_meta,
            **metrics_naive, **metrics_meta
        })


if __name__ == "__main__":
    main()
