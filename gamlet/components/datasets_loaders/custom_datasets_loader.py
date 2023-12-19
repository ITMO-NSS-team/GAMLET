from __future__ import annotations

from typing import Callable

from gamlet.data_preparation.dataset import CustomDataset, DatasetIDType
from gamlet.components.datasets_loaders import DatasetsLoader


class CustomDatasetsLoader(DatasetsLoader):
    dataset_class = CustomDataset

    def __init__(self,
                 dataset_from_id_func: Callable[[DatasetIDType], CustomDataset] = CustomDataset):
        self.dataset_from_id_func = dataset_from_id_func

    def load_single(self, dataset_id: DatasetIDType) -> CustomDataset:
        dataset = self.dataset_from_id_func(dataset_id)
        return dataset
