import random

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from sklearn import preprocessing


class DataLoader:
    def __init__(self, datasets: dict, split_ratio: float = 0.7):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        self.split_ratio = split_ratio

    def get_data(self, dataset_name: str = None):
        if dataset_name is None:
            dataset_name = random.choice(self.dataset_names)

        path = self.datasets[dataset_name]

        return self.load_data(path)

    def load_data(self, path_to_dataset):
        data = InputData.from_csv(path_to_dataset)

        train_data, test_data = train_test_data_setup(data=data, split_ratio=self.split_ratio)

        return train_data, test_data


class DataLoader_TS(DataLoader):
    def __init__(self, datasets: dict, forecast_length: dict = None, path_to_meta_data: str = None):
        super().__init__(datasets)
        self.dataset_name = None
        self.forecast_length = None
        self.forecast_length_mapping = {'D': 14, 'W': 13, 'M': 18, 'Q': 8, 'Y': 6}
        self.meta_data = pd.read_csv(path_to_meta_data, index_col=0) if path_to_meta_data else None
        self.meta_data = self.normilize_data(self.meta_data).transpose()

    def get_data(self, dataset_name: str = None):
        if dataset_name is None:
            self.dataset_name = random.choice(self.dataset_names)
        else:
            self.dataset_name = dataset_name

        path = self.datasets[self.dataset_name]
        ts_type = self.dataset_name[3]
        self.forecast_length = self.forecast_length_mapping[ts_type]

        if isinstance(self.meta_data, pd.DataFrame):
            meta_data = self.meta_data[self.dataset_name].values
        else:
            meta_data = None

        return self.load_data(path, meta_data)

    def load_data(self, path_to_dataset, meta_data):
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(self.forecast_length))
        data = pd.read_csv(path_to_dataset)['value'].values.reshape(-1, 1)

        X = data[:-self.forecast_length]
        y = data[-self.forecast_length:]

        train_data = InputData(
            idx=np.arange(len(X)),
            features=np.ravel(X),
            target=np.ravel(X),
            task=task,
            data_type=DataTypesEnum.ts
        )

        start_forecast = len(train_data.features)
        end_forecast = start_forecast + self.forecast_length
        test_data = InputData(
            idx=np.arange(start_forecast, end_forecast),
            features=np.ravel(X),
            target=np.ravel(y),
            task=task,
            data_type=DataTypesEnum.ts
        )

        return train_data, test_data, meta_data

    def normilize_data(self, data):
        x = data.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        return pd.DataFrame(x_scaled, index=data.index, columns=data.columns)
