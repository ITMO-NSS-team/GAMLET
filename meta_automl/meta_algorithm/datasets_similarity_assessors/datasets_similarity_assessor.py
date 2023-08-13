from abc import abstractmethod
from typing import List


class DatasetsSimilarityAssessor:
    """The Datasets similarity assessor is designed to assess the similarity of datasets by meta-features.

    At the knowledge acquisition stage, a table of meta-features of datasets is input.
    At the knowledge application stage, the component accepts meta-signs of new datasets and converts them into
    lists of similar datasets from the previously "memorized" ones.

    Optionally, it can also return a measure of similarity (or distance) between
    the new datasets and the "memorized" ones.
    """

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
