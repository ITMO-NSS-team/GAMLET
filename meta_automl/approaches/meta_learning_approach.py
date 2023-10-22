from abc import ABC, abstractmethod
from dataclasses import dataclass


class MetaLearningApproach(ABC):
    @dataclass
    class Parameters:
        any_param: None

    @dataclass
    class Data:
        pass

    @dataclass
    class Components:
        pass

    def __init__(self, *args, **kwargs):
        self.parameters = self.Parameters(*args, **kwargs)
        self.data = self.Data()
        self.components = self.Components()

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError()
