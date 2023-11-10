from meta_automl.data_preparation.dataset import DatasetIDType, TimeSeriesDataset
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader


class TimeSeriesDatasetsLoader(DatasetsLoader):
    dataset_class = TimeSeriesDataset

    def __init__(self, forecast_length: int = 1, custom_path=None):
        super().__init__()
        self.dataset_ids = set()
        self.forecast_length = forecast_length
        self.custom_path = custom_path

    def load_single(self, dataset_id: DatasetIDType) -> TimeSeriesDataset:
        dataset = TimeSeriesDataset(dataset_id, self.forecast_length, self.custom_path)
        self.dataset_ids.add(dataset.id_)
        return dataset
