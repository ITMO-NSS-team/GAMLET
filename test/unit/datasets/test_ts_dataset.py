from pathlib import Path

from meta_automl.data_preparation.dataset import TimeSeriesData, TimeSeriesDataset
from meta_automl.data_preparation.file_system import get_project_root


def test_ts_dataset_creation(timeseries_dataset_ids):
    for dataset_id in timeseries_dataset_ids:
        dataset = TimeSeriesDataset(dataset_id)

        assert dataset.id_ == dataset_id


def test_ts_dataset_data_loading(timeseries_dataset_ids):
    for dataset_id in timeseries_dataset_ids:
        dataset = TimeSeriesDataset(dataset_id,
                                    custom_path=Path(get_project_root(), 'test', 'data', 'cache', 'datasets',
                                                     'custom_dataset'))
        dataset_data = dataset.get_data()
        assert isinstance(dataset_data, TimeSeriesData)
