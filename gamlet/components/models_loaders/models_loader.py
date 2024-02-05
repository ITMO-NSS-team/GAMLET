from abc import ABC, abstractmethod
from typing import List

from gamlet.data_preparation.evaluated_model import EvaluatedModel


class ModelsLoader(ABC):

    @abstractmethod
    def load(self, *args, **kwargs) -> List[List[EvaluatedModel]]:
        raise NotImplementedError()
