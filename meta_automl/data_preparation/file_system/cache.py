from __future__ import annotations

import pickle
from pathlib import Path

from typing import Type, Any, Dict, TYPE_CHECKING

import openml

from meta_automl.data_preparation.file_system.cache_properties import CacheProperties, CacheType
from meta_automl.data_preparation.file_system.file_system import get_data_dir, ensure_dir_exists

if TYPE_CHECKING:
    from meta_automl.data_preparation.dataset import DatasetBase
    from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor


class CacheOperator:
    pass


def get_openml_cache_dir() -> Path:
    return get_data_dir().joinpath('openml_cache')


def get_full_openml_cache_dir() -> Path:
    return get_data_dir().joinpath('openml_cache/org/openml/www')


def update_openml_cache_dir():
    openml_cache_path = str(get_openml_cache_dir())
    openml.config.set_cache_directory(openml_cache_path)


def _get_cache_path(object_class: Type[CacheOperator], object_id: str, _create_parent_dir: bool = True) -> Path:
    cache_properties = get_cache_properties(object_class.__name__)
    directory = cache_properties.dir_
    path = cache_properties.template.format(id_=object_id)
    path = directory.joinpath(path)
    if _create_parent_dir:
        ensure_dir_exists(directory)
    return path


def get_dataset_cache_path(dataset: DatasetBase) -> Path:
    class_ = dataset.__class__
    id_ = dataset.id_
    return _get_cache_path(class_, str(id_))


def get_dataset_cache_path_by_id(class_: Type[DatasetBase], id_: Any) -> Path:
    return _get_cache_path(class_, str(id_))


def get_meta_features_cache_path(extractor_class: Type[MetaFeaturesExtractor], dataset_id: Any) -> Path:
    return _get_cache_path(extractor_class, str(dataset_id))


def get_local_meta_features(extractor_class: Type[MetaFeaturesExtractor], dataset_id: Any) -> Dict[str, Any]:
    meta_features_file = get_meta_features_cache_path(extractor_class, dataset_id)
    if not meta_features_file.exists():
        return {}
    with open(meta_features_file, 'rb') as f:
        meta_features = pickle.load(f)
    return meta_features


def update_local_meta_features(extractor_class: Type[MetaFeaturesExtractor],
                               dataset_id: Any, meta_features: Dict[str, Any]):
    meta_features_file = get_meta_features_cache_path(extractor_class, dataset_id)
    meta_features_old = get_local_meta_features(extractor_class, dataset_id)
    with open(meta_features_file, 'wb') as f:
        meta_features_old.update(meta_features)
        pickle.dump(meta_features_old, f)


def get_cache_properties(class_name: str) -> CacheProperties:
    cache_properties_by_class_name = {
        'OpenMLDataset': CacheProperties(
            type_=CacheType.directory,
            dir_=get_full_openml_cache_dir().joinpath('datasets'),
            template='{id_}'),
        'CustomDataset': CacheProperties(
            type_=CacheType.file,
            dir_=get_data_dir().joinpath('datasets/custom_dataset'),
            template='{id_}.pkl'),
        'PymfeExtractor': CacheProperties(
            type_=CacheType.file,
            dir_=get_data_dir().joinpath('metafeatures/pymfe'),
            template='{id_}.pkl'),
    }
    try:
        return cache_properties_by_class_name[class_name]
    except KeyError as e:
        raise KeyError(f'Cache properties for the class {class_name} are not defined.').with_traceback(e.__traceback__)
