from __future__ import annotations

from typing import List, Sequence, Callable

from meta_automl.data_preparation.dataset import DatasetIDType, CustomDataset, \
    DatasetBase
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader


class CustomDatasetsLoader(DatasetsLoader):
    def __init__(self,
                 dataset_from_id_func: Callable[[DatasetIDType], DatasetBase] = CustomDataset):
        self.dataset_ids = set()
        self.dataset_from_id_func = dataset_from_id_func

    def load(self, dataset_ids: Sequence[DatasetIDType]) -> List[CustomDataset]:
        datasets = []
        for dataset_id in dataset_ids:
            dataset = self.load_single(dataset_id)
            datasets.append(dataset)
        return datasets

    def load_single(self, dataset_id: DatasetIDType) -> CustomDataset:
        dataset = self.dataset_from_id_func(dataset_id)
        self.dataset_ids.add(dataset.id_)

        return dataset
