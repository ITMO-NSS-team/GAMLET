from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING, Type

import openml

from meta_automl.data_preparation.file_system.cache_properties import CacheProperties, CacheType
from meta_automl.data_preparation.file_system.file_system import ensure_dir_exists, get_data_dir

if TYPE_CHECKING:
    from meta_automl.data_preparation.dataset import DatasetBase


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
    id_ = dataset.id
    return _get_cache_path(class_, str(id_))


def get_dataset_cache_path_by_id(class_: Type[DatasetBase], id_: Any) -> Path:
    return _get_cache_path(class_, str(id_))


def get_cache_properties(class_name: str) -> CacheProperties:
    cache_properties_by_class_name = {
        'OpenMLDataset': CacheProperties(
            type=CacheType.file,
            dir=get_openml_cache_dir() / 'datasets',
            path_template='{id}/dataset.arff'),
        'CustomDataset': CacheProperties(
            type=CacheType.file,
            dir=get_cache_dir() / 'datasets/custom_dataset',
            path_template='{id}.pkl')
    }
    try:
        return cache_properties_by_class_name[class_name]
    except KeyError as e:
        raise KeyError(f'Cache properties for the class {class_name} are not defined.').with_traceback(e.__traceback__)
