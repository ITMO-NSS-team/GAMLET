from abc import ABC, abstractmethod
from dataclasses import dataclass


class MetaLearningApproach(ABC):
    """ This is an abstract class of a meta learning approach. Subclasses must implement the method `predict()`
    and may implement the following classes and corresponding fields: `Parameters`, `Data`, `Components`.
    These classes can be used to define field sets with expected field types for the meta learning approach.
    It is recommended to initialize these classes into `__init__()` method and save them into instance fields
    before applying the approach.
    """

    @dataclass
    class Parameters:
        """ It aims to define and store all parameters of the approach.
        It is recommended to fill into `__init__()`. """
        pass

    @dataclass
    class Data:
        """ It aims to define and store training data needed for this approach.
        It is recommended to fill into `fit()`. """
        pass

    @dataclass
    class Components:
        """ It aims to define and store instances of the meta-learning components needed for this approach.
        It is recommended to fill into `__init__()`"""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """ Here you can implement your approach's logic with the usage of data, parameters and components. """
        raise NotImplementedError()
