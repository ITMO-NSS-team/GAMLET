from pathlib import Path
from typing import Union

from meta_automl.data_preparation.dataset import Dataset, DatasetCache
from test.constants import CACHED_DATASETS
from test.data_manager import TestDataManager


def assert_file_unmodified_during_test(path: Path, test_start_timestamp: float):
    assert path.stat().st_mtime < test_start_timestamp, f'The file should not be modified during the test: ' \
                                                        f'"{path.relative_to(TestDataManager.get_project_root())}".'


def assert_cache_file_exists(path: Path):
    assert path.exists(), 'Cache not found at the path: ' \
                          f'"{path.relative_to(TestDataManager.get_project_root())}".'


def check_dataset_and_cache(dataset_or_cache: Union[Dataset, DatasetCache], desired_name: str, desired_path: Path,
                            test_start_time: float):
    assert dataset_or_cache.name == desired_name
    assert dataset_or_cache.cache_path == desired_path
    assert_cache_file_exists(desired_path)
    if desired_name in CACHED_DATASETS:
        assert_file_unmodified_during_test(desired_path, test_start_time)
