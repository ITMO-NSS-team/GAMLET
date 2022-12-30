from abc import abstractmethod

from typing import List


class DatasetsSimilarityAssessor:
    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    @abstractmethod
    def datasets(self) -> List[str]:
        raise NotImplementedError()
