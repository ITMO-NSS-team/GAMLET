from __future__ import annotations

from typing import List

from golem.core.log import default_log

from meta_automl.data_preparation.dataset import OpenMLDataset, OpenMLDatasetIDType
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader


class OpenMLDatasetsLoader(DatasetsLoader):
    def __init__(self):
        self.dataset_ids = []

    def load(self, dataset_ids: List[OpenMLDatasetIDType]) -> List[OpenMLDataset]:
        self.dataset_ids = dataset_ids

        datasets = []
        # TODO: Optimize like this
        #  https://github.com/openml/automlbenchmark/commit/a09dc8aee96178dd14837d9e1cd519d1ec63f804
        for dataset_id in self.dataset_ids:
            dataset = self.load_single(dataset_id)
            datasets.append(dataset)
        return datasets

    def load_single(self, dataset_id: OpenMLDatasetIDType) -> OpenMLDataset:
        return OpenMLDataset(dataset_id)

    @property
    def _log(self):
        return default_log(self)
