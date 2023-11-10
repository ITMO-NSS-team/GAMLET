from pathlib import Path

from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.file_system import get_cache_dir, get_data_dir, get_dataset_cache_path_by_id
from meta_automl.data_preparation.file_system.cache import (get_dataset_cache_path, get_local_meta_features,
                                                            get_meta_features_cache_path, get_openml_cache_dir)
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from test.constants import DATASETS_WITH_CACHED_META_FEATURES, OPENML_CACHED_DATASETS

CACHED_LOCAL_META_FEATURES = {'attr_to_inst': 0.020289855072463767, 'cat_to_num': 1.3333333333333333,
                              'freq_class.mean': 0.5, 'freq_class.sd': 0.07788422517417042,
                              'inst_to_attr': 49.285714285714285, 'nr_attr': 14, 'nr_bin': 4, 'nr_cat': 8,
                              'nr_class': 2, 'nr_inst': 690, 'nr_num': 6, 'num_to_cat': 0.75}


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


def test_get_meta_features_cache_path():
    cache_dir = get_cache_dir()
    id_ = DATASETS_WITH_CACHED_META_FEATURES[0]
    class_ = OpenMLDataset.__name__
    cache_path = get_meta_features_cache_path(PymfeExtractor, OpenMLDataset, id_)
    relative_path = cache_path.relative_to(cache_dir)
    assert relative_path == Path(f'metafeatures/pymfe/{class_}_{id_}.pkl')


def test_get_local_meta_features():
    id_ = DATASETS_WITH_CACHED_META_FEATURES[0]
    local_meta_features = get_local_meta_features(PymfeExtractor, OpenMLDataset, id_)
    assert local_meta_features == CACHED_LOCAL_META_FEATURES
