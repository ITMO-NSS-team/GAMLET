from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING, Type

import openml

from meta_automl.data_preparation.file_system.cache_properties import CacheProperties, CacheType
from meta_automl.data_preparation.file_system.file_system import ensure_dir_exists, get_data_dir

if TYPE_CHECKING:
    from meta_automl.data_preparation.dataset import DatasetBase
    from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor


class CacheOperator:
    pass


def get_cache_dir() -> Path:
    return ensure_dir_exists(get_data_dir().joinpath('cache'))


def get_openml_cache_dir() -> Path:
    return Path(openml.config.get_cache_directory())


def update_openml_cache_dir():
    openml_cache_path = get_cache_dir().joinpath('openml_cache')
    openml.config.set_root_cache_directory(str(openml_cache_path))


def _get_cache_path(object_class: Type[CacheOperator], object_id: str, _create_parent_dir: bool = True,
                    **path_kwargs) -> Path:
    cache_properties = get_cache_properties(object_class.__name__)
    directory = cache_properties.dir
    path = cache_properties.path_template.format(id=object_id, **path_kwargs)
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


def get_meta_features_cache_path(extractor_class: Type[MetaFeaturesExtractor], dataset_class: Type[DatasetBase],
                                 dataset_id: Any) -> Path:
    return _get_cache_path(extractor_class, str(dataset_id), dataset_class=dataset_class.__name__)


def get_local_meta_features(extractor_class: Type[MetaFeaturesExtractor], dataset_class: Type[DatasetBase],
                            dataset_id: Any) -> Dict[str, Any]:
    meta_features_file = get_meta_features_cache_path(extractor_class, dataset_class, dataset_id)
    if not meta_features_file.exists():
        return {}
    with open(meta_features_file, 'rb') as f:
        meta_features = pickle.load(f)
    return meta_features


def update_local_meta_features(extractor_class: Type[MetaFeaturesExtractor], dataset_class: Type[DatasetBase],
                               dataset_id: Any, meta_features: Dict[str, Any]):
    meta_features_file = get_meta_features_cache_path(extractor_class, dataset_class, dataset_id)
    meta_features_old = get_local_meta_features(extractor_class, dataset_class, dataset_id)
    with open(meta_features_file, 'wb') as f:
        meta_features_old.update(meta_features)
        pickle.dump(meta_features_old, f)


def get_cache_properties(class_name: str) -> CacheProperties:
    cache_properties_by_class_name = {
        'OpenMLDataset': CacheProperties(
            type=CacheType.file,
            dir=get_openml_cache_dir().joinpath('datasets'),
            path_template='{id}/dataset.arff'),
        'CustomDataset': CacheProperties(
            type=CacheType.file,
            dir=get_cache_dir().joinpath('datasets/custom_dataset'),
            path_template='{id}.pkl'),
        'PymfeExtractor': CacheProperties(
            type=CacheType.file,
            dir=get_cache_dir().joinpath('metafeatures/pymfe'),
            path_template='{dataset_class}_{id}.pkl'),
        'TimeSeriesFeaturesExtractor': CacheProperties(
            type=CacheType.file,
            dir=get_cache_dir().joinpath('metafeatures/tsfe'),
            path_template='{id}.pkl')
    }
    try:
        return cache_properties_by_class_name[class_name]
    except KeyError as e:
        raise KeyError(f'Cache properties for the class {class_name} are not defined.').with_traceback(e.__traceback__)
