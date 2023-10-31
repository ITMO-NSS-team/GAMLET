from abc import ABC, abstractmethod
from typing import List

from meta_automl.data_preparation.evaluated_model import EvaluatedModel


class ModelsLoader(ABC):

    @abstractmethod
    def load(self, *args, **kwargs) -> List[List[EvaluatedModel]]:
        raise NotImplementedError()
