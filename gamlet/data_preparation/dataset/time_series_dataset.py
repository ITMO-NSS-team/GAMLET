from pathlib import Path

import pandas as pd

from gamlet.data_preparation.dataset import DatasetIDType, TimeSeriesData
from gamlet.data_preparation.dataset.dataset_base import DatasetBase
from gamlet.data_preparation.file_system import get_project_root


class TimeSeriesDataset(DatasetBase):

    def __init__(self, id_: DatasetIDType, forecast_length: int = 1, custom_path=None):
        super().__init__(id_)
        self.forecast_length = forecast_length
        if custom_path:
            self.path_to_knowledge_base = Path(custom_path)
        else:
            self.path_to_knowledge_base = Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets')

    def get_data(self) -> TimeSeriesData:
        path_to_series = Path(self.path_to_knowledge_base, self.id, 'data.csv')
        series = pd.read_csv(path_to_series)['value'].values
        x = series[:-self.forecast_length]
        y = series[-self.forecast_length:]
        return TimeSeriesData(self, x, y, self.forecast_length)
