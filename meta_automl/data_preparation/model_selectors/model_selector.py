from abc import abstractmethod


class ModelSelector:

    @abstractmethod
    def select(self, *args, **kwargs):
        raise NotImplementedError()
