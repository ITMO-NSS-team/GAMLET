from abc import abstractmethod
from pathlib import Path
from typing import List

from support.data_utils import get_openml_dataset


class DatasetsLoader:
    def __init__(self, datasets=None):
        self.datasets = datasets or []

    def __call__(self, *args, **kwargs) -> List[Path]:
        if not self.datasets:
            raise ValueError('No data sources provided!')
        return self._get_datasets()

    @abstractmethod
    def _get_datasets(self) -> List[Path]:
        raise NotImplementedError()


class OpenmlLoader(DatasetsLoader):
    def _get_datasets(self) -> List[Path]:
        datasets = []
        for source in self.datasets:
            path = get_openml_dataset(source)
            datasets.append(path)
        self.datasets = datasets
        return self.datasets
