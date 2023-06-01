from __future__ import annotations

import pickle
from pathlib import Path

from meta_automl.data_preparation.data_manager import DataManager
from meta_automl.data_preparation.dataset import DatasetBase
from meta_automl.data_preparation.dataset.dataset_base import DatasetData


class DataNotFoundError(FileNotFoundError):
    pass


class CustomDataset(DatasetBase):
    source_name = 'file_dataset'

    @property
    def cache_path(self) -> Path:
        return DataManager.get_dataset_cache_path(self.id_, self.source_name)

    def get_data(self) -> DatasetData:
        if not self.cache_path.exists():
            raise DataNotFoundError(f'Dataset {self} is missing by the path "{self.cache_path}".')
        with open(self.cache_path, 'rb') as f:
            dataset_data = pickle.load(f)
        return dataset_data

    def dump_data(self, dataset_data: DatasetData, cache_path: Path) -> CustomDataset:
        cache_path = cache_path or self.cache_path
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset_data, f)
        return self
