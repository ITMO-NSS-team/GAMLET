from __future__ import annotations

from typing import TypeVar, Union

import openml

from meta_automl.data_preparation.dataset import DatasetBase, TabularData
from meta_automl.data_preparation.file_system import update_openml_cache_dir

OpenMLDatasetIDType = TypeVar('OpenMLDatasetIDType', bound=int)

update_openml_cache_dir()


class OpenMLDataset(DatasetBase):

    def __init__(self, id_: OpenMLDatasetIDType):
        if isinstance(id_, str):
            raise ValueError('Creating OpenMLDataset by dataset name is ambiguous. Please, use dataset id.'
                             f'Otherwise, you can perform search by f{self.__class__.__name__}.from_search().')
        self._openml_dataset = openml.datasets.get_dataset(id_, download_data=True, download_qualities=False,
                                                           download_features_meta_data=False,
                                                           error_if_multiple=True)
        id_ = self._openml_dataset.id
        name = self._openml_dataset.name
        super().__init__(id_, name)

    @classmethod
    def from_search(cls, id_: Union[OpenMLDatasetIDType, str], **get_dataset_kwargs) -> OpenMLDataset:
        openml_dataset = openml.datasets.get_dataset(id_, download_data=True, download_qualities=False,
                                                     download_features_meta_data=False,
                                                     **get_dataset_kwargs)
        return cls(openml_dataset.id)

    def get_data(self) -> TabularData:
        X, y, categorical_indicator, attribute_names = self._openml_dataset.get_data(
            target=self._openml_dataset.default_target_attribute
        )
        X = X.to_numpy()
        y = y.to_numpy()
        return TabularData(self, X, y, categorical_indicator, attribute_names)
