from __future__ import annotations

from typing import List, Union

import openml

from meta_automl.data_preparation.data_directory_manager import DataDirectoryManager
from meta_automl.data_preparation.dataset import DatasetCache, Dataset
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader


OpenMLDatasetID = Union[str, int]


class OpenMLDatasetsLoader(DatasetsLoader):
    data_manager = DataDirectoryManager

    def __init__(self):
        self.dataset_sources = []

    def load(self, dataset_sources: List[OpenMLDatasetID]) -> List[DatasetCache]:
        self.dataset_sources = dataset_sources

        datasets = []
        # TODO: Optimize like this
        #  https://github.com/openml/automlbenchmark/commit/a09dc8aee96178dd14837d9e1cd519d1ec63f804
        for source in self.dataset_sources:
            dataset = self.load_single(source)
            datasets.append(dataset)
        return datasets

    def load_single(self, source: OpenMLDatasetID):
        return self.get_openml_dataset(source)

    def get_openml_dataset(self, dataset_id: OpenMLDatasetID, force_download: bool = False) -> DatasetCache:
        openml_dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
        name = openml_dataset.name.lower()
        dataset_cache_path = self.data_manager.get_dataset_cache_path(name)
        if dataset_cache_path.exists() and not force_download:
            dataset_cache = DatasetCache(name, dataset_cache_path)
        else:
            dataset_id = openml_dataset.id
            X, y, categorical_indicator, attribute_names = openml_dataset.get_data(
                target=openml_dataset.default_target_attribute,
                dataset_format='array'
            )
            dataset = Dataset(name, X, y, categorical_indicator, attribute_names, _id=dataset_id)
            dataset_cache = dataset.dump_to_cache(dataset_cache_path)
        return dataset_cache
