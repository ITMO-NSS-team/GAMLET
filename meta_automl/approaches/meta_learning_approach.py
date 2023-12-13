from abc import ABC, abstractmethod
from dataclasses import dataclass


class MetaLearningApproach(ABC):
    @dataclass
    class Parameters:
        pass

    @dataclass
    class Data:
        pass

    @dataclass
    class Components:
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError()
