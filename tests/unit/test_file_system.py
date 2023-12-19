from pathlib import Path

from gamlet.data_preparation.file_system import get_data_dir, get_project_root
from gamlet.data_preparation.file_system.file_system import get_checkpoints_dir


def test_root_dir():
    project_root = get_project_root()
    relative_path = Path(__file__).relative_to(project_root)
    assert relative_path == Path('tests/unit/test_file_system.py')


def test_data_dir():
    project_root = get_project_root()
    data_dir = get_data_dir()
    relative_path = data_dir.relative_to(project_root)
    assert relative_path == Path('tests/data')


def test_checkpoints_dir():
    project_root = get_project_root()
    checkpoints_dir = get_checkpoints_dir()
    relative_path = checkpoints_dir.relative_to(project_root)
    assert relative_path == Path('model_checkpoints')
