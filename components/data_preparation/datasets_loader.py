from __future__ import annotations

from abc import abstractmethod
from typing import List, Union

import openml

from components.data_preparation.dataset import DatasetCache, Dataset
from support.data_utils import get_dataset_cache_path

OpenMLDatasetID = Union[str, int]


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


class OpenMLDatasetsLoader(DatasetsLoader):
    def __init__(self):
        self.dataset_sources = []

    def fit(self, dataset_sources: List[str]):
        self.dataset_sources = dataset_sources
        return self

    def load(self) -> List[DatasetCache]:
        if not self.dataset_sources:
            raise ValueError('No data sources provided!')

        datasets = []
        # TODO: Optimize like this
        #  https://github.com/openml/automlbenchmark/commit/a09dc8aee96178dd14837d9e1cd519d1ec63f804
        for source in self.dataset_sources:
            dataset = self.load_single(source)
            datasets.append(dataset)
        return datasets

    def load_single(self, source):
        return self.get_openml_dataset(source)

    @staticmethod
    def get_openml_dataset(dataset_id: OpenMLDatasetID) -> DatasetCache:
        openml_dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
        name = openml_dataset.name
        dataset_cache_path = get_dataset_cache_path(name)
        if dataset_cache_path.exists():
            dataset_cache = DatasetCache(name, dataset_cache_path)
        else:
            X, y, categorical_indicator, attribute_names = openml_dataset.get_data(
                target=openml_dataset.default_target_attribute,
                dataset_format='array'
            )
            dataset_cache = Dataset(name, X, y, categorical_indicator, attribute_names).dump(dataset_cache_path)
        return dataset_cache
