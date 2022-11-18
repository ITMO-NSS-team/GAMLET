from __future__ import annotations

import pickle
from os import PathLike
from pathlib import Path
from typing import Dict, Any, Union

import openml

from components.data_preparation.dataset import Dataset, DatasetCache

PathType = Union[PathLike, str]
OpenMLDatasetId = Union[str, int]


def get_openml_dataset(dataset_id: OpenMLDatasetId) -> DatasetCache:
    openml_dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
    name = openml_dataset.name
    dataset_cache_path = get_dataset_cache_path(name)
    if dataset_cache_path.exists():
        dataset_cache = DatasetCache(name, dataset_cache_path)
    else:
        X, y, categorical_indicator, attribute_names = openml_dataset.get_data(
            target=openml_dataset.default_target_attribute,
            dataset_format='array'
        )
        dataset_cache = Dataset(name, X, y, categorical_indicator, attribute_names).dump(dataset_cache_path)
    return dataset_cache


def get_dataset_cache_path(dataset_name: str) -> Path:
    return get_dataset_dir(dataset_name).joinpath(dataset_name).with_suffix(DatasetCache.default_cache_extension)


def get_dataset_dir(dataset_name: str) -> Path:
    return ensure_dir_exists(get_datasets_dir().joinpath(dataset_name))


def get_datasets_dir() -> Path:
    datasets_dir = get_data_dir().joinpath('datasets')
    return ensure_dir_exists(datasets_dir)


def get_data_dir() -> Path:
    data_dir = project_root().joinpath('data')
    return ensure_dir_exists(data_dir)


def ensure_dir_exists(dir_: Path) -> Path:
    if not dir_.exists():
        dir_.mkdir()
    return dir_


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def get_meta_features_cache_path(dataset_name: str, source_name: str):
    dataset_dir = get_dataset_dir(dataset_name)
    return dataset_dir.joinpath(source_name).with_suffix('.pkl')


def get_meta_features_dict(dataset_name: str, source_name: str) -> Dict[str, Any]:
    meta_features_file = get_meta_features_cache_path(dataset_name, source_name)
    if not meta_features_file.exists():
        return {}
    with open(meta_features_file, 'rb') as f:
        meta_features = pickle.load(f)
    return meta_features


def update_meta_features_dict(dataset_name: str, source_name: str, meta_features: Dict[str, Any]):
    meta_features_file = get_meta_features_cache_path(dataset_name, source_name)
    meta_features_old = get_meta_features_dict(dataset_name, source_name)
    with open(meta_features_file, 'wb') as f:
        meta_features_old.update(meta_features)
        pickle.dump(meta_features, f)
