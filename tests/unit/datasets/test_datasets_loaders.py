from gamlet.components.datasets_loaders import OpenMLDatasetsLoader, TimeSeriesDatasetsLoader
from tests.unit.datasets.general_checks import check_dataset_cache


def test_group_load_new_datasets(openml_dataset_ids):
    loader = OpenMLDatasetsLoader()
    datasets = loader.load(openml_dataset_ids)
    assert set(d.id for d in datasets) == set(openml_dataset_ids)
    for dataset_id, dataset in zip(openml_dataset_ids, datasets):
        check_dataset_cache(dataset)


def test_load_single(openml_dataset_ids):
    loader = OpenMLDatasetsLoader()
    for dataset_id in openml_dataset_ids:
        dataset = loader.load_single(dataset_id)
        check_dataset_cache(dataset)


def test_load_new_datasets_on_demand(openml_dataset_ids):
    loader = OpenMLDatasetsLoader()
    for dataset_id in openml_dataset_ids:
        dataset = loader.load_single(dataset_id)
        check_dataset_cache(dataset)


def test_group_load_new_datasets_ts(timeseries_dataset_ids):
    loader = TimeSeriesDatasetsLoader()
    datasets = loader.load(timeseries_dataset_ids)
    assert set(d.id for d in datasets) == set(timeseries_dataset_ids)


def test_load_single_ts(timeseries_dataset_ids):
    loader = TimeSeriesDatasetsLoader()
    for dataset_id in timeseries_dataset_ids:
        loader.load_single(dataset_id)
