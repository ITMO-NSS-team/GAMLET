from pathlib import Path

from gamlet.data_preparation.dataset import OpenMLDataset
from gamlet.data_preparation.file_system import get_cache_dir, get_data_dir, get_dataset_cache_path_by_id
from gamlet.data_preparation.file_system.cache import get_dataset_cache_path, get_openml_cache_dir
from test.constants import OPENML_CACHED_DATASETS


def test_cache_dir():
    data_dir = get_data_dir()
    cache_dir = get_cache_dir()
    relative_path = cache_dir.relative_to(data_dir)
    assert relative_path == Path('cache')


def test_get_openml_cache_dir():
    cache_dir = get_cache_dir()
    openml_cache_dir = get_openml_cache_dir()
    relative_path = openml_cache_dir.relative_to(cache_dir)
    assert relative_path == Path('openml_cache/org/openml/www')


def test_get_dataset_cache_path():
    openml_cache_dir = get_openml_cache_dir()
    id_ = OPENML_CACHED_DATASETS[0]
    dataset = OpenMLDataset(id_)
    cache_path = get_dataset_cache_path(dataset)
    relative_path = cache_path.relative_to(openml_cache_dir)
    assert relative_path == Path(f'datasets/{id_}/dataset.arff')


def test_get_dataset_cache_path_by_id():
    openml_cache_dir = get_openml_cache_dir()
    id_ = OPENML_CACHED_DATASETS[0]
    cache_path = get_dataset_cache_path_by_id(OpenMLDataset, id_)
    relative_path = cache_path.relative_to(openml_cache_dir)
    assert relative_path == Path(f'datasets/{id_}/dataset.arff')


def test_dataset_cache_path():
    openml_cache_dir = get_openml_cache_dir()
    id_ = OPENML_CACHED_DATASETS[0]
    dataset = OpenMLDataset(id_)
    cache_path = dataset.cache_path
    relative_path = cache_path.relative_to(openml_cache_dir)
    assert relative_path == Path(f'datasets/{id_}/dataset.arff')
