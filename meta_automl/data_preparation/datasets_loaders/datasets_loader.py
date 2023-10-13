from __future__ import annotations

from abc import abstractmethod
from typing import List

from meta_automl.data_preparation.dataset import DatasetBase


class DatasetsLoader:

    @abstractmethod
    def load(self, *args, **kwargs) -> List[DatasetBase]:
        raise NotImplementedError()

    @abstractmethod
    def load_single(self, *args, **kwargs) -> DatasetBase:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dataset_class(self):
        """ Should be implemented as a class field or property with corresponding name. """
        pass
