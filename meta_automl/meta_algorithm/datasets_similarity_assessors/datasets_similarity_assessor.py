from abc import abstractmethod


class DatasetsSimilarityAssessor:
    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_similar(self, *args, **kwargs):
        raise NotImplementedError()
