import time
from pathlib import Path
from typing import Union

import pytest

from meta_automl.data_preparation.dataset import DatasetCache, Dataset
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from test.data_manager import TestDataManager

PRELOADED_DATASET_NAMES = ['australian']


@pytest.fixture(scope='module')
def dataset_names():
    dataset_names = ['australian', 'monks-problems-1', 'blood-transfusion-service-center']
    yield dataset_names
    for name in PRELOADED_DATASET_NAMES:
        dataset_names.remove(name)
    for dataset_name in dataset_names:
        TestDataManager.get_dataset_cache_path(dataset_name).unlink()


def test_group_load_new_datasets(dataset_names):
    test_time_start = time.time()
    loader = OpenMLDatasetsLoader()
    loader.data_manager = TestDataManager

    datasets = loader.load(dataset_names)

    assert loader.dataset_sources == dataset_names

    for dataset_name, dataset_cache in zip(dataset_names, datasets):
        check_dataset_and_cache(dataset_cache, dataset_name, dataset_cache.cache_path, test_time_start)


def test_load_single(dataset_names):
    test_time_start = time.time()
    loader = OpenMLDatasetsLoader()
    loader.data_manager = TestDataManager
    for dataset_name in dataset_names:
        dataset_cache = loader.load_single(dataset_name)
        check_dataset_and_cache(dataset_cache, dataset_name, dataset_cache.cache_path, test_time_start)


def test_load_new_datasets_on_demand(dataset_names):
    test_time_start = time.time()
    loader = OpenMLDatasetsLoader()
    loader.data_manager = TestDataManager
    for dataset_name in dataset_names:
        cache_path = TestDataManager.get_dataset_cache_path(dataset_name)
        dataset = loader.cache_to_memory(DatasetCache(dataset_name, cache_path))
        check_dataset_and_cache(dataset, dataset_name, cache_path, test_time_start)


def check_dataset_and_cache(dataset_or_cache: Union[Dataset, DatasetCache], desired_name: str, desired_path: Path,
                            test_time_start: float):
    assert desired_path.exists()
    assert dataset_or_cache.name == desired_name
    assert dataset_or_cache.cache_path == desired_path
    if desired_name in PRELOADED_DATASET_NAMES:
        assert desired_path.stat().st_mtime < test_time_start, \
            'Pre-loaded cache should not be modified during the test.'
