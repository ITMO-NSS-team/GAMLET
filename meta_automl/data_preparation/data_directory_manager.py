from __future__ import annotations

import pickle
from os import PathLike
from pathlib import Path
from typing import Dict, Any, Union

PathType = Union[PathLike, str]
DEFAULT_CACHE_EXTENSION = '.pkl'


class DataDirectoryManager:

    @classmethod
    def get_dataset_cache_path(cls, dataset_name: str) -> Path:
        return cls.get_datasets_dir().joinpath(dataset_name).with_suffix(DEFAULT_CACHE_EXTENSION)

    @classmethod
    def get_datasets_dir(cls) -> Path:
        datasets_dir = cls.get_data_dir().joinpath('datasets')
        return cls.ensure_dir_exists(datasets_dir)

    @classmethod
    def get_data_dir(cls) -> Path:
        data_dir = cls.get_project_root().joinpath('data')
        return cls.ensure_dir_exists(data_dir)

    @classmethod
    def ensure_dir_exists(cls, dir_: Path) -> Path:
        if not dir_.exists():
            dir_.mkdir()
        return dir_

    @classmethod
    def get_project_root(cls) -> Path:
        """Returns project root folder."""
        return Path(__file__).parents[2]

    @classmethod
    def get_meta_features_cache_path(cls, dataset_name: str, source_name: str):
        meta_features_dir = cls.ensure_dir_exists(cls.get_data_dir().joinpath(source_name))
        return meta_features_dir.joinpath(dataset_name).with_suffix('.pkl')

    @classmethod
    def get_meta_features_dict(cls, dataset_name: str, source_name: str) -> Dict[str, Any]:
        meta_features_file = cls.get_meta_features_cache_path(dataset_name, source_name)
        if not meta_features_file.exists():
            return {}
        with open(meta_features_file, 'rb') as f:
            meta_features = pickle.load(f)
        return meta_features

    @classmethod
    def update_meta_features_dict(cls, dataset_name: str, source_name: str, meta_features: Dict[str, Any]):
        meta_features_file = cls.get_meta_features_cache_path(dataset_name, source_name)
        meta_features_old = cls.get_meta_features_dict(dataset_name, source_name)
        with open(meta_features_file, 'wb') as f:
            meta_features_old.update(meta_features)
            pickle.dump(meta_features, f)
