from pathlib import Path

import test.constants
from meta_automl.data_preparation.dataset import DatasetBase
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.file_system import get_dataset_cache_path


def assert_file_unmodified_during_test(path: Path):
    failure_message = ('The file should not be modified during the test: '
                       f'"{path.relative_to(get_project_root())}".')
    assert path.stat().st_mtime < test.constants.TEST_START_TIMESTAMP, failure_message


def assert_cache_file_exists(path: Path):
    assert path.exists(), 'Cache not found at the path: ' \
                          f'"{path.relative_to(get_project_root())}".'


def check_dataset_cache(dataset: DatasetBase):
    cache_path = get_dataset_cache_path(dataset)
    assert_cache_file_exists(cache_path)
    if dataset.id_ in test.constants.OPENML_CACHED_DATASETS:
        assert_file_unmodified_during_test(cache_path)
