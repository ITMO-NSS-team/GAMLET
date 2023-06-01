from __future__ import annotations

from abc import abstractmethod
from typing import List, Type

from meta_automl.data_preparation.data_manager import DataManager
from meta_automl.data_preparation.dataset import DatasetBase


class DatasetsLoader:
    data_manager: Type[DataManager] = DataManager

    @abstractmethod
    def load(self, *args, **kwargs) -> List[DatasetBase]:
        raise NotImplementedError()

    @abstractmethod
    def load_single(self, *args, **kwargs) -> DatasetBase:
        raise NotImplementedError()
