from __future__ import annotations

import openml

from meta_automl.data_preparation.dataset import DatasetBase
from meta_automl.data_preparation.dataset.dataset_base import DatasetData
from meta_automl.data_preparation.file_system import update_openml_cache_dir

OpenMLDatasetIDType = int

update_openml_cache_dir()


class OpenMLDataset(DatasetBase):

    def __init__(self, id_: OpenMLDatasetIDType):
        self._openml_dataset = openml.datasets.get_dataset(id_, download_data=False, download_qualities=False)
        name = self._openml_dataset.name
        super().__init__(id_, name)

    def get_data(self, dataset_format: str = 'dataframe') -> DatasetData:
        X, y, categorical_indicator, attribute_names = self._openml_dataset.get_data(
            target=self._openml_dataset.default_target_attribute,
            dataset_format=dataset_format
        )
        return DatasetData(X, y, categorical_indicator, attribute_names)
