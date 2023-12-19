from gamlet.data_preparation.dataset import DatasetData, OpenMLDataset
from gamlet.data_preparation.file_system import get_dataset_cache_path_by_id
from test.constants import OPENML_CACHED_DATASETS
from test.unit.datasets.general_checks import check_dataset_cache


def test_openml_dataset_cache_exists_only_if_preloaded(openml_dataset_ids):
    for dataset_id in openml_dataset_ids:
        cache_path = get_dataset_cache_path_by_id(OpenMLDataset, dataset_id)

        is_exist = dataset_id in OPENML_CACHED_DATASETS
        assert is_exist == cache_path.exists()


def test_openml_dataset_creation(openml_dataset_ids):
    for dataset_id in openml_dataset_ids:
        dataset = OpenMLDataset(dataset_id)

        assert dataset.id == dataset_id


def test_openml_dataset_data_loading(openml_dataset_ids):
    for dataset_id in openml_dataset_ids:
        dataset = OpenMLDataset(dataset_id)
        dataset_data = dataset.get_data()
        assert isinstance(dataset_data, DatasetData)
        check_dataset_cache(dataset)
