from __future__ import annotations

from abc import abstractmethod
from typing import List, Sequence

from meta_automl.data_preparation.dataset import DatasetBase, DatasetIDType


class DatasetsLoader:

    def load(self, dataset_ids: Sequence[DatasetIDType]) -> List[DatasetBase]:
        datasets = []
        for dataset_id in dataset_ids:
            dataset = self.load_single(dataset_id)
            datasets.append(dataset)
        return datasets

    @abstractmethod
    def load_single(self, *args, **kwargs) -> DatasetBase:
        raise NotImplementedError()
