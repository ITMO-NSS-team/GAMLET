from abc import abstractmethod


class ModelSelector:

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def select(self, *args, **kwargs):
        raise NotImplementedError()
