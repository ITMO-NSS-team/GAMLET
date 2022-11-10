from __future__ import annotations

from abc import abstractmethod
from typing import List, TYPE_CHECKING

from support.data_utils import get_openml_dataset

if TYPE_CHECKING:
    from components.dataset import Dataset


class DatasetsLoader:
    def __init__(self, dataset_sources=None):
        self.dataset_sources = dataset_sources or []
        self.datasets = []

    def __call__(self) -> List[Dataset]:
        if not self.dataset_sources:
            raise ValueError('No data sources provided!')
        return self._get_datasets()

    @abstractmethod
    def _get_datasets(self) -> List[Dataset]:
        raise NotImplementedError()


class OpenmlLoader(DatasetsLoader):
    def _get_datasets(self) -> List[Dataset]:
        datasets = []
        for source in self.dataset_sources:
            dataset = get_openml_dataset(source)
            datasets.append(dataset)
        self.datasets = datasets
        return self.datasets
