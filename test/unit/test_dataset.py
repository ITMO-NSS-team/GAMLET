import numpy as np
import pytest

from meta_automl.data_preparation.dataset import DatasetCache, NoCacheError
from test.data_manager import TestDataManager


@pytest.fixture
def dumped_cache_path():
    path = TestDataManager.get_dataset_cache_path('data_dumped')
    yield path
    path.unlink()


def test_dataset_caching(dumped_cache_path):
    dataset_name = 'australian'

    cache_path = TestDataManager.get_dataset_cache_path(dataset_name)

    dataset_cache = DatasetCache(dataset_name, cache_path)
    dataset = dataset_cache.from_cache()
    dumped_cache = dataset.dump_to_cache(dumped_cache_path)
    reloaded_dataset = dumped_cache.from_cache()
    # Check data integrity.
    assert dataset.name == dataset_name
    assert reloaded_dataset.name == dataset_name
    assert dataset.id == reloaded_dataset.id
    assert np.all(np.equal(dataset.X, reloaded_dataset.X))
    assert np.all(np.equal(dataset.y, reloaded_dataset.y))
    # Check caching integrity.
    assert dataset_cache.cache_path == cache_path
    assert dataset.cache_path == cache_path
    assert dumped_cache.cache_path == dumped_cache_path
    assert reloaded_dataset.cache_path == dumped_cache_path


def test_error_on_missing_dataset_cache():
    with pytest.raises(NoCacheError):
        DatasetCache('random_missing_cache').from_cache()
