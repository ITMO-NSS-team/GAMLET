from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, List, Sequence, TypeVar

from meta_automl.data_preparation.dataset import DatasetBase, DatasetIDType

DatasetType_co = TypeVar('DatasetType_co', bound=DatasetBase, covariant=True)


class DatasetsLoader(Generic[DatasetType_co], ABC):

    def load(self, dataset_ids: Sequence[DatasetIDType]) -> List[DatasetType_co]:
        datasets = []
        for dataset_id in dataset_ids:
            dataset = self.load_single(dataset_id)
            datasets.append(dataset)
        return datasets

    @abstractmethod
    def load_single(self, *args, **kwargs) -> DatasetType_co:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dataset_class(self):
        """ Should be implemented as a class field or property with corresponding name. """
        pass
