from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Hashable, List, Optional, TypeVar

import numpy as np

from gamlet.data_preparation.file_system import CacheOperator, get_dataset_cache_path

DatasetType_co = TypeVar('DatasetType_co', bound='DatasetBase', covariant=True)


@dataclass
class DatasetData(Generic[DatasetType_co], Sequence):
    dataset: DatasetType_co

    x: np.ndarray
    y: Optional[np.ndarray] = None

    @property
    def id(self):
        return self.dataset.id

    def __getitem__(self, item):
        other = copy(self)
        if self.y is not None:
            other.y = self.y[item]
        other.x = self.x[item]
        return other

    def __len__(self):
        return len(self.x)


@dataclass
class TabularData(DatasetData):
    categorical_indicator: Optional[List[bool]] = None
    attribute_names: Optional[List[str]] = None


@dataclass
class TimeSeriesData(DatasetData):
    # time series has already split
    forecast_length: int = 1


DatasetDataType_co = TypeVar('DatasetDataType_co', bound=DatasetData, covariant=True)

DatasetIDType = TypeVar('DatasetIDType', bound=Hashable)


class DatasetBase(Generic[DatasetDataType_co], CacheOperator, ABC):

    def __init__(self, id_: DatasetIDType, name: Optional[str] = None):
        self.id = id_
        self.name = name

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id}, name={self.name})'

    @abstractmethod
    def get_data(self) -> DatasetDataType_co:
        raise NotImplementedError()

    @property
    def cache_path(self) -> Path:
        return get_dataset_cache_path(self)
