from __future__ import annotations

from abc import abstractmethod
from typing import List

from meta_automl.data_preparation.dataset import DatasetCache


class DatasetsLoader:

    @abstractmethod
    def fit(self, *args, **kwargs) -> DatasetsLoader:
        raise NotImplementedError()

    @abstractmethod
    def load(self, *args, **kwargs) -> List[DatasetCache]:
        raise NotImplementedError()

    @abstractmethod
    def load_single(self, *args, **kwargs) -> DatasetCache:
        raise NotImplementedError()
