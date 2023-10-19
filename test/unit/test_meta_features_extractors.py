import shutil
from pathlib import Path

import pytest

from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.dataset.time_series_dataset import TimeSeriesDataset
from meta_automl.data_preparation.file_system import get_dataset_cache_path_by_id, get_meta_features_cache_path, \
    get_project_root
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.meta_features_extractors.time_series.time_series_meta_features_extractor import \
    TimeSeriesFeaturesExtractor
from test.constants import OPENML_DATASET_IDS_TO_LOAD, OPENML_CACHED_DATASETS, DATASETS_WITH_CACHED_META_FEATURES, \
    TS_DATASETS_WITH_CACHED_META_FEATURES, TS_DATASETS_IDS_TO_LOAD
from test.unit.datasets.general_checks import assert_file_unmodified_during_test, assert_cache_file_exists


@pytest.fixture
def dataset_ids():
    dataset_ids = list(set(OPENML_CACHED_DATASETS + DATASETS_WITH_CACHED_META_FEATURES + OPENML_DATASET_IDS_TO_LOAD))
    yield dataset_ids
    for dataset_id in dataset_ids:
        if dataset_id not in OPENML_CACHED_DATASETS:
            dataset_cache_path = get_dataset_cache_path_by_id(OpenMLDataset, dataset_id)
            shutil.rmtree(dataset_cache_path.parent)
        if dataset_id not in DATASETS_WITH_CACHED_META_FEATURES:
            mf_cache_path = get_meta_features_cache_path(PymfeExtractor, OpenMLDataset, dataset_id)
            mf_cache_path.unlink(missing_ok=True)


@pytest.fixture
def timeseries_dataset_ids():
    ids = TS_DATASETS_IDS_TO_LOAD
    yield ids


def test_table_meta_features_extraction(dataset_ids):
    extractor = PymfeExtractor(extractor_params={'groups': 'general'})
    meta_features = extractor.extract(dataset_ids)
    assert list(meta_features.index) == dataset_ids
    for dataset_id in dataset_ids:
        meta_features_cache_path = get_meta_features_cache_path(PymfeExtractor, OpenMLDataset, dataset_id)
        assert_cache_file_exists(meta_features_cache_path)

        if dataset_id in DATASETS_WITH_CACHED_META_FEATURES:
            assert_file_unmodified_during_test(meta_features_cache_path)
        else:
            cache_path = get_dataset_cache_path_by_id(OpenMLDataset, dataset_id)
            assert_cache_file_exists(cache_path)


def test_ts_meta_features_extraction(timeseries_dataset_ids):
    extractor = TimeSeriesFeaturesExtractor(custom_path=Path(get_project_root(), 'test', 'data', 'cache', 'datasets',
                                                     'custom_dataset'))
    meta_features = extractor.extract(timeseries_dataset_ids, use_cached=False)
    assert list(meta_features.index) == timeseries_dataset_ids
    for dataset_id in timeseries_dataset_ids:
        meta_features_cache_path = get_meta_features_cache_path(TimeSeriesFeaturesExtractor, TimeSeriesDataset,
                                                                dataset_id)
        assert_cache_file_exists(meta_features_cache_path)

        if dataset_id in TS_DATASETS_WITH_CACHED_META_FEATURES:
            assert_file_unmodified_during_test(meta_features_cache_path)
        else:
            cache_path = get_dataset_cache_path_by_id(TimeSeriesDataset, dataset_id)
            assert_cache_file_exists(cache_path)
