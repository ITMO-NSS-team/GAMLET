from __future__ import annotations

import openml

from meta_automl.data_preparation.data_manager import DataManager
from meta_automl.data_preparation.dataset import DatasetBase
from meta_automl.data_preparation.dataset.dataset_base import DatasetData

OpenMLDatasetIDType = int
openml_cache_path = str(DataManager.get_data_dir().joinpath('openml_cache'))
openml.config.set_cache_directory(openml_cache_path)


class OpenMLDataset(DatasetBase):
    source_name = 'openml'

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
