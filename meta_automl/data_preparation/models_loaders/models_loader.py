from abc import abstractmethod

from typing import List

from meta_automl.data_preparation.model import Model


class ModelsLoader:

    @abstractmethod
    def load(self, *args, **kwargs) -> List[List[Model]]:
        raise NotImplementedError()
