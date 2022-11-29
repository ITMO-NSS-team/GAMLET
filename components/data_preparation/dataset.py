from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import scipy as sp


@dataclass
class DatasetCache:
    default_cache_extension = '.pkl'

    name: str
    cache_path: Path

    def load(self) -> Dataset:
        with open(self.cache_path, 'rb') as f:
            dataset = pickle.load(f)
        dataset.cache_path = self.cache_path
        return dataset


@dataclass
class Dataset:
    name: str
    X: Union[np.ndarray, pd.DataFrame, sp.sparse.csr_matrix]
    y: Optional[Union[np.ndarray, pd.DataFrame]] = None
    categorical_indicator: Optional[List[bool]] = None
    attribute_names: Optional[List[str]] = None
    cache_path: Optional[Path] = None

    def dump(self, cache_path: Optional[Path] = None) -> DatasetCache:
        cache_path = cache_path or self.cache_path
        self.cache_path = cache_path
        with open(cache_path, 'wb') as f:
            pickle.dump(self, f)
        return DatasetCache(self.name, cache_path)
