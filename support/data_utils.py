from __future__ import annotations

import pickle
from os import PathLike
from pathlib import Path
from typing import Dict, Any, Union

PathType = Union[PathLike, str]
DEFAULT_CACHE_EXTENSION = '.pkl'


def get_dataset_cache_path(dataset_name: str) -> Path:
    return get_datasets_dir().joinpath(dataset_name).with_suffix(DEFAULT_CACHE_EXTENSION)


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
    meta_features_dir = ensure_dir_exists(get_data_dir().joinpath(source_name))
    return meta_features_dir.joinpath(dataset_name).with_suffix('.pkl')


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
