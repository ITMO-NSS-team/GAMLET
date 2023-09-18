import os
import shutil
from pathlib import Path
from secrets import token_hex
import pandas as pd
from meta_automl.data_preparation.dataset import DatasetIDType
from meta_automl.data_preparation.dataset.dataset_base import TimeSeriesData, DatasetBase
from meta_automl.data_preparation.file_system import get_project_root


class TimeSeriesDataset(DatasetBase):
    path_to_knowledge_base = Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets')

    def __init__(self, id_: DatasetIDType, forecast_length: int = 1):
        super().__init__(id_)
        self.forecast_length = forecast_length

    def get_data(self) -> TimeSeriesData:
        path_to_series = Path(self.path_to_knowledge_base, self.id_, 'data', f'{self.id_}.csv')
        series = pd.read_csv(path_to_series)['value'].values
        x = series[:-self.forecast_length],
        y = series[-self.forecast_length:]
        return TimeSeriesData(x, y, self.forecast_length)


if __name__ == '__main__':

    p = r'D:\Programming\Work\pytsbe\experiments\univariate\benchmark_results_Daily\M4_daily_test\launch_0\FEDOT'

    # for i in os.listdir(Path(get_project_root(), r'data/knowledge_base_time_series_0/datasets')):
    #     os.mkdir(Path(get_project_root(), r'data/knowledge_base_time_series_0/datasets', i.split('.')[0], 'models', 'FEDOT'))
    #
    #
    for i in os.listdir(p):
        if 'additional' in i:
            label = 'M4_' + i.split('_')[0]
            n_p = Path(p, i)
            s = token_hex(16)
            shutil.copyfile(Path(n_p, 'opt_history.json'),
                            Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets', label,
                                 'histories', 'FEDOT', s + '.json'))
            shutil.copyfile(Path(n_p, 'model', '0_pipeline_saved', '0_pipeline_saved.json'),
                            Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets', label,
                                 'models', 'FEDOT', s + '.json'))
