import pandas as pd
import pytest

from gamlet.data_preparation.dataset import CustomDataset, DataNotFoundError, TabularData
from tests.unit.datasets.general_checks import assert_cache_file_exists


def get_new_dataset_data(dataset) -> TabularData:
    dataset_data = TabularData(
        dataset=dataset,
        x=pd.DataFrame([['a', 'b'], ['b', 'a']]),
        y=pd.DataFrame([[5], [10]]),
        categorical_indicator=[True, True],
        attribute_names=['foo', 'bar']
    )
    return dataset_data


@pytest.fixture(scope='module')
def new_dataset():
    dataset = CustomDataset(42, 'custom_dataset_for_test')
    new_dataset_data = get_new_dataset_data(dataset)
    dataset.dump_data(new_dataset_data)
    yield dataset
    dataset.cache_path.unlink()


def test_error_on_missing_dataset_cache():
    with pytest.raises(DataNotFoundError):
        CustomDataset('random_missing_dataset').get_data()


def test_custom_dataset_dumping(new_dataset):
    # Act
    cache_path = new_dataset.cache_path
    # Assert
    assert_cache_file_exists(cache_path)


def test_custom_dataset_data_loading(new_dataset):
    # Act
    correct_data = get_new_dataset_data(new_dataset)
    dataset = new_dataset
    data = dataset.get_data()
    # Assert
    assert tuple(vars(data)) == tuple(vars(correct_data))
