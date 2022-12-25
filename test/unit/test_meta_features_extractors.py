import time

import pytest

from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from test.general_checks import assert_file_unmodified_during_test, assert_cache_file_exists
from test.data_manager import TestDataManager
from test.constants import CACHED_DATASETS, DATASETS_WITH_CACHED_META_FEATURES


@pytest.fixture
def dataset_names():
    dataset_names = ['australian', 'monks-problems-1', 'monks-problems-2', 'blood-transfusion-service-center']
    yield dataset_names
    for dataset_name in dataset_names:
        if dataset_name not in CACHED_DATASETS + DATASETS_WITH_CACHED_META_FEATURES:
            TestDataManager.get_dataset_cache_path(dataset_name).unlink(missing_ok=True)
        if dataset_name not in DATASETS_WITH_CACHED_META_FEATURES:
            TestDataManager.get_meta_features_cache_path(dataset_name, PymfeExtractor.SOURCE).unlink(missing_ok=True)


def test_meta_features_extraction(dataset_names):
    test_start_time = time.time()
    extractor = PymfeExtractor(extractor_params={'groups': 'general'})
    extractor.data_manager = TestDataManager
    extractor.datasets_loader.data_manager = TestDataManager
    meta_features = extractor.extract(dataset_names)
    assert list(meta_features.index) == dataset_names
    for dataset_name in dataset_names:
        meta_features_cache_path = TestDataManager.get_meta_features_cache_path(
            dataset_name, extractor.SOURCE)
        assert_cache_file_exists(meta_features_cache_path)

        if dataset_name in DATASETS_WITH_CACHED_META_FEATURES:
            assert_file_unmodified_during_test(meta_features_cache_path, test_start_time)
        else:
            cache_path = TestDataManager.get_dataset_cache_path(dataset_name)
            assert_cache_file_exists(cache_path)
