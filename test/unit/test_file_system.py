from pathlib import Path

from meta_automl.data_preparation.file_system import get_cache_dir, get_data_dir, get_project_root


def test_root_dir():
    project_root = get_project_root()
    relative_path = Path(__file__).relative_to(project_root)
    assert relative_path == Path('test/unit/test_file_system.py')


def test_data_dir():
    project_root = get_project_root()
    data_dir = get_data_dir()
    relative_path = data_dir.relative_to(project_root)
    assert relative_path == Path('test/data')
