from __future__ import annotations

from abc import abstractmethod
from typing import List

from meta_automl.data_preparation.dataset import Dataset, DatasetCache, NoCacheError


class DatasetsLoader:

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
            return self.load_single(dataset.name).from_cache()
