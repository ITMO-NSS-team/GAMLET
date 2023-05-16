"""Adapted from https://github.com/aimclub/FEDOT/blob/84fd5a60f207c75a6af9a38af3bac7c85a9a4252/examples/advanced/surrogate_optimization.py"""

from abc import abstractmethod
from typing import Any

from golem.core.dag.graph import Graph


class SurrogateModel:
    """
    Model for evaluating fitness function without time-consuming fitting pipeline
    """

    @abstractmethod
    def __call__(self, graph: Graph, **kwargs: Any):
        raise NotImplementedError()
