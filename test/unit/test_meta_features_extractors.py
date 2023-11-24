import shutil
from pathlib import Path

import pytest

from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.datasets_loaders import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.file_system import get_dataset_cache_path_by_id, get_project_root
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor, TimeSeriesFeaturesExtractor
from test.constants import OPENML_CACHED_DATASETS, OPENML_DATASET_IDS_TO_LOAD, TS_DATASETS_IDS_TO_LOAD
from test.unit.datasets.general_checks import assert_cache_file_exists


@pytest.fixture
def dataset_ids():
    dataset_ids = list(set(OPENML_CACHED_DATASETS + OPENML_DATASET_IDS_TO_LOAD))
    yield dataset_ids
    for dataset_id in dataset_ids:
        if dataset_id not in OPENML_CACHED_DATASETS:
            dataset_cache_path = get_dataset_cache_path_by_id(OpenMLDataset, dataset_id)
            shutil.rmtree(dataset_cache_path.parent, ignore_errors=True)


@pytest.fixture
def timeseries_dataset_ids():
    ids = TS_DATASETS_IDS_TO_LOAD
    yield ids


def test_table_meta_features_extraction(dataset_ids):
    extractor = PymfeExtractor(extractor_params={'groups': 'general'})
    datasets = [OpenMLDataset(dataset_id) for dataset_id in dataset_ids]
    meta_features = extractor.extract(datasets, fill_input_nans=True)
    assert list(meta_features.index) == dataset_ids
    for dataset_id in dataset_ids:
        dataset_cache_path = get_dataset_cache_path_by_id(OpenMLDataset, dataset_id)
        assert_cache_file_exists(dataset_cache_path)  # Extractor downloaded necessary data


def test_ts_meta_features_extraction(timeseries_dataset_ids):
    datasets_loader = TimeSeriesDatasetsLoader(
        custom_path=Path(get_project_root(),
                         'test', 'data', 'cache', 'datasets', 'custom_dataset'))
    datasets = datasets_loader.load(dataset_ids=timeseries_dataset_ids)
    extractor = TimeSeriesFeaturesExtractor()
    meta_features = extractor.extract(datasets)
    assert list(meta_features.index) == timeseries_dataset_ids
