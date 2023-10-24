import time
from pathlib import Path

import pytest

from meta_automl.data_preparation.file_system import file_system, get_data_dir, get_project_root
from meta_automl.data_preparation.file_system import update_openml_cache_dir
from test import constants


def pytest_configure():
    # Crucial setup & checks to avoid misplacing data during the tests
    check_project_root()
    set_data_dir()
    check_data_dir()
    update_openml_cache_dir()


def check_project_root():
    actual_root = Path(__file__).parents[1]
    root = get_project_root()
    if root != actual_root:
        pytest.exit(f'The function `get_project_root()` should point to "{actual_root}". '
                    f'Got "{root}" instead', 1)


def set_data_dir():
    file_system.DATA_SUBDIR = constants.TEST_DATA_SUBDIR


def check_data_dir():
    data_dir = get_data_dir()
    if data_dir.relative_to(get_project_root()) != Path(constants.TEST_DATA_SUBDIR):
        pytest.exit(f'The function `get_data_dir()` should point to "test/data" (relative to project root). '
                    f'Got "{data_dir}" instead', 1)


@pytest.fixture(scope="session", autouse=True)
def set_test_start_timestamp():
    constants.TEST_START_TIMESTAMP = time.time()
