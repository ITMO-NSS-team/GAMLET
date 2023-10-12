from meta_automl.data_preparation.dataset import DatasetIDType
from meta_automl.data_preparation.dataset.time_series_dataset import TimeSeriesDataset
from meta_automl.data_preparation.datasets_loaders.custom_datasets_loader import CustomDatasetsLoader


class TimeSeriesDatasetsLoader(CustomDatasetsLoader):
    def __init__(self, forecast_length: int = 1):
        super().__init__()
        self.dataset_ids = set()
        self.forecast_length = forecast_length

    def load_single(self, dataset_id: DatasetIDType) -> TimeSeriesDataset:
        dataset = TimeSeriesDataset(dataset_id, self.forecast_length)
        self.dataset_ids.add(dataset.id_)
        return dataset
