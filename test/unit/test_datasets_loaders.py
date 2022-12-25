import time

import pytest

from meta_automl.data_preparation.dataset import DatasetCache
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from test.general_checks import check_dataset_and_cache
from test.constants import CACHED_DATASETS
from test.data_manager import TestDataManager


@pytest.fixture
def dataset_names():
    dataset_names = ['australian', 'blood-transfusion-service-center']
    yield dataset_names
    for dataset_name in dataset_names:
        if dataset_name not in CACHED_DATASETS:
            TestDataManager.get_dataset_cache_path(dataset_name).unlink(missing_ok=True)


def test_group_load_new_datasets(dataset_names):
    test_start_time = time.time()
    loader = OpenMLDatasetsLoader()
    loader.data_manager = TestDataManager

    datasets = loader.load(dataset_names)

    assert loader.dataset_sources == dataset_names

    for dataset_name, dataset_cache in zip(dataset_names, datasets):
        check_dataset_and_cache(dataset_cache, dataset_name, dataset_cache.cache_path, test_start_time)


def test_load_single(dataset_names):
    test_start_time = time.time()
    loader = OpenMLDatasetsLoader()
    loader.data_manager = TestDataManager
    for dataset_name in dataset_names:
        dataset_cache = loader.load_single(dataset_name)
        check_dataset_and_cache(dataset_cache, dataset_name, dataset_cache.cache_path, test_start_time)


def test_load_new_datasets_on_demand(dataset_names):
    test_start_time = time.time()
    loader = OpenMLDatasetsLoader()
    loader.data_manager = TestDataManager
    for dataset_name in dataset_names:
        cache_path = TestDataManager.get_dataset_cache_path(dataset_name)
        dataset = loader.cache_to_memory(DatasetCache(dataset_name, cache_path))
        check_dataset_and_cache(dataset, dataset_name, cache_path, test_start_time)
