from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import scipy as sp

from meta_automl.data_preparation.data_manager import DataManager


class NoCacheError(FileNotFoundError):
    pass


@dataclass
class DatasetCache:
    name: str
    _cache_path: Optional[Path] = None
    _id: Optional[int] = None

    @property
    def id(self):
        return self._id or self.name

    @property
    def cache_path(self):
        return self._cache_path or DataManager.get_dataset_cache_path(self.name)

    @cache_path.setter
    def cache_path(self, val):
        self._cache_path = val

    def from_cache(self) -> Dataset:
        if not self.cache_path.exists():
            raise NoCacheError(f'Dataset {self.name} not found!')
        with open(self.cache_path, 'rb') as f:
            dataset = pickle.load(f)
        dataset.cache_path = self.cache_path
        return dataset


@dataclass
class Dataset:
    name: str
    x: Union[np.ndarray, pd.DataFrame, sp.sparse.csr_matrix]
    y: Optional[Union[np.ndarray, pd.DataFrame]] = None
    categorical_indicator: Optional[List[bool]] = None
    attribute_names: Optional[List[str]] = None
    cache_path: Optional[Path] = None
    _id: Optional[int] = None

    def dump_to_cache(self, cache_path: Optional[Path] = None) -> DatasetCache:
        cache_path = cache_path or self.cache_path
        with open(cache_path, 'wb') as f:
            pickle.dump(self, f)
        return DatasetCache(self.name, cache_path, self.id)

    @property
    def id(self):
        return self._id or self.name
