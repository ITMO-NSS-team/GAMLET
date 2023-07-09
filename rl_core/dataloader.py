import random

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup


class DataLoader:
    def __init__(self, datasets: dict, split_ratio: float = 0.7):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        self.split_ratio = split_ratio

    def get_data(self, dataset_name: str = None):
        if dataset_name is None:
            dataset_name = random.choice(self.dataset_names)

        path = self.datasets[dataset_name]
        data = InputData.from_csv(path)

        train_data, test_data = train_test_data_setup(data=data, split_ratio=self.split_ratio)

        return train_data, test_data


