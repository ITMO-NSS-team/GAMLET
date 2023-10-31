from abc import ABC, abstractmethod
from typing import List

from meta_automl.data_preparation.model import Model


class ModelsLoader(ABC):

    @abstractmethod
    def load(self, *args, **kwargs) -> List[List[Model]]:
        raise NotImplementedError()
