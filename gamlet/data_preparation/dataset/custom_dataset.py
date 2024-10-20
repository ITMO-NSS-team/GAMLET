from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from gamlet.data_preparation.dataset import DatasetBase
from gamlet.data_preparation.dataset.dataset_base import DatasetDataType_co


class DataNotFoundError(FileNotFoundError):
    pass


class CustomDataset(DatasetBase):

    def get_data(self, cache_path: Optional[Path] = None) -> DatasetDataType_co:
        cache_path = cache_path or self.cache_path
        if not cache_path.exists():
            raise DataNotFoundError(f'Dataset {self} is missing by the path "{cache_path}".')
        with open(cache_path, 'rb') as f:
            dataset_data = pickle.load(f)
        return dataset_data

    def dump_data(self, dataset_data: DatasetDataType_co, cache_path: Optional[Path] = None) -> CustomDataset:
        cache_path = cache_path or self.cache_path
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset_data, f)
        return self
