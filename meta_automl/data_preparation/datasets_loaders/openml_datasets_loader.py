from __future__ import annotations

from typing import List, Optional, Sequence, Union

from meta_automl.data_preparation.dataset import OpenMLDataset, OpenMLDatasetIDType
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader


class OpenMLDatasetsLoader(DatasetsLoader):
    dataset_class = OpenMLDataset

    def __init__(self, allow_names: bool = False):
        self._allow_names = allow_names

    def load(self, dataset_ids: Sequence[Union[OpenMLDatasetIDType, str]],
             allow_names: Optional[bool] = None) -> List[OpenMLDataset]:
        allow_names = self._allow_names if allow_names is None else allow_names

        datasets = []
        # TODO: Optimize like this
        #  https://github.com/openml/automlbenchmark/commit/a09dc8aee96178dd14837d9e1cd519d1ec63f804
        for dataset_id in dataset_ids:
            dataset = self.load_single(dataset_id, allow_name=allow_names)
            datasets.append(dataset)
        return datasets

    def load_single(self, dataset_id: Union[OpenMLDatasetIDType, str],
                    allow_name: Optional[bool] = None) -> OpenMLDataset:
        allow_name = self._allow_names if allow_name is None else allow_name

        if allow_name:
            dataset = OpenMLDataset.from_search(dataset_id)
        else:
            dataset = OpenMLDataset(dataset_id)

        return dataset
