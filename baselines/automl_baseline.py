from abc import ABC


class AutoMLBaseline(ABC):
    def run(self):
        raise NotImplementedError

    @staticmethod
    def save_on_disk(data):
        raise NotImplementedError

