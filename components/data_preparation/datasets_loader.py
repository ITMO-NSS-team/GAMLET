from __future__ import annotations

from abc import abstractmethod
from typing import List, TYPE_CHECKING

from support.data_utils import get_openml_dataset

if TYPE_CHECKING:
    from components.data_preparation.dataset import DatasetCache


class DatasetsLoader:
    def __init__(self, dataset_sources=None):
        self.dataset_sources = dataset_sources or []
        self.datasets: List[DatasetCache] = []

    @abstractmethod
    def load(self, *args, **kwargs) -> List[DatasetCache]:
        raise NotImplementedError()


class OpenMLDatasetsLoader(DatasetsLoader):
    def load(self) -> List[DatasetCache]:
        if not self.dataset_sources:
            raise ValueError('No data sources provided!')

        datasets = []
        # TODO: Optimize like this
        #  https://github.com/openml/automlbenchmark/commit/a09dc8aee96178dd14837d9e1cd519d1ec63f804
        for source in self.dataset_sources:
            dataset = get_openml_dataset(source)
            datasets.append(dataset)
        self.datasets = datasets
        return self.datasets
