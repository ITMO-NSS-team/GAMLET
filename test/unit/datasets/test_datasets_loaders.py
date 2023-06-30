from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from test.unit.datasets.general_checks import check_dataset_cache


def test_group_load_new_datasets(openml_dataset_ids):
    loader = OpenMLDatasetsLoader()
    datasets = loader.load(openml_dataset_ids)
    assert loader.dataset_ids == openml_dataset_ids
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