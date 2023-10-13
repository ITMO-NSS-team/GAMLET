import shutil

import pytest

from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.file_system import get_dataset_cache_path_by_id
from test.constants import OPENML_CACHED_DATASETS, OPENML_DATASET_IDS_TO_LOAD, TS_DATASETS_IDS_TO_LOAD


@pytest.fixture
def openml_dataset_ids():
    ids = OPENML_DATASET_IDS_TO_LOAD
    yield ids
    for dataset_id in ids:
        if dataset_id in OPENML_CACHED_DATASETS:
            continue
        cache_path = get_dataset_cache_path_by_id(OpenMLDataset, dataset_id)
        shutil.rmtree(cache_path.parent, ignore_errors=True)


@pytest.fixture
def timeseries_dataset_ids():
    ids = TS_DATASETS_IDS_TO_LOAD
    yield ids

