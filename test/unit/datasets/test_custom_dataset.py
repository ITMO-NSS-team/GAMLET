import numpy as np
import pandas as pd
import pytest

from meta_automl.data_preparation.dataset import DataNotFoundError, CustomDataset, DatasetData
from test.unit.datasets.general_checks import assert_cache_file_exists


@pytest.fixture(scope='module')
def new_dataset_data():
    dataset_data = DatasetData(
        x=pd.DataFrame([['a', 'b'], ['b', 'a']]),
        y=pd.DataFrame([[5], [10]]),
        categorical_indicator=[True, True],
        attribute_names=['foo', 'bar']
    )
    return dataset_data


@pytest.fixture(scope='module')
def new_dataset(new_dataset_data):
    dataset = CustomDataset(42, 'custom_dataset_for_test')
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


def test_custom_dataset_data_loading(new_dataset_data, new_dataset):
    # Act
    correct_data = new_dataset_data
    dataset = new_dataset
    data = dataset.get_data()
    # Assert
    assert np.all(np.equal(data.x, correct_data.x))
    assert np.all(np.equal(data.y, correct_data.y))
    assert data.categorical_indicator == correct_data.categorical_indicator
    assert data.attribute_names == correct_data.attribute_names
