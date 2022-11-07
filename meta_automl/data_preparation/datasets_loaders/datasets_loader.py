from __future__ import annotations

from abc import abstractmethod
from typing import List, Type

from meta_automl.data_preparation.data_manager import DataManager
from meta_automl.data_preparation.dataset import Dataset, DatasetCache, NoCacheError


class DatasetsLoader:
    data_manager: Type[DataManager] = DataManager

    @abstractmethod
    def load(self, *args, **kwargs) -> List[DatasetCache]:
        raise NotImplementedError()

    @abstractmethod
    def load_single(self, *args, **kwargs) -> DatasetCache:
        raise NotImplementedError()

    def cache_to_memory(self, dataset: DatasetCache) -> Dataset:
        try:
            return dataset.from_cache()
        except NoCacheError:
            return self.load_single(dataset.id).from_cache()
