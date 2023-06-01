from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Union, Optional, List, Any

import numpy as np
import pandas as pd
import scipy as sp


DatasetIDType = Any


@dataclass
class DatasetData:
    x: Union[np.ndarray, pd.DataFrame, sp.sparse.csr_matrix]
    y: Optional[Union[np.ndarray, pd.DataFrame]] = None
    categorical_indicator: Optional[List[bool]] = None
    attribute_names: Optional[List[str]] = None


class DatasetBase(ABC):
    source_name: str

    def __init__(self, id_: DatasetIDType, name: Optional[str] = None):
        self.id_ = id_
        self.name = name

    def __repr__(self):
        return f'{self.__class__.__name__}(id_={self.id_}, name={self.name})'

    @abstractmethod
    def get_data(self) -> DatasetData:
        raise NotImplementedError()
