from __future__ import annotations

from pathlib import Path

import openml
import pandas as pd
from scipy.io.arff import loadarff

from components.dataset import Dataset, DatasetCache


def get_openml_dataset(dataset_id) -> DatasetCache:
    openml_dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
    name = openml_dataset.name
    dataset_cache_path = get_dataset_cache_path(name)
    if dataset_cache_path.exists():
        dataset_cache = DatasetCache(name, dataset_cache_path)
        # data = pd.read_csv(dataset_cache_path)
        # X = data.drop('target', axis='columns')
        # y = data[['target']]
    else:
        X, y, categorical_indicator, attribute_names = openml_dataset.get_data(
            target=openml_dataset.default_target_attribute,
            dataset_format='array'
        )
        dataset_cache = Dataset(name, X, y, categorical_indicator, attribute_names).dump(dataset_cache_path)
        # data = pd.DataFrame(X, columns=attribute_names)
        # data['target'] = y
        # data.to_csv(dataset_cache_path)
    # dataset = Dataset(name, X, y, dataset_cache_path)
    return dataset_cache


def arff_to_csv(input_path, output_path):
    raw_data = loadarff(input_path)
    df = pd.DataFrame(raw_data[0])
    df.to_csv(output_path)


def get_dataset_cache_path(dataset_name: str) -> Path:
    return get_dataset_dir(dataset_name).joinpath(dataset_name).with_suffix(DatasetCache.default_cache_extension)


def get_dataset_dir(dataset_name: str) -> Path:
    return ensure_dir_exists(get_datasets_dir().joinpath(dataset_name))


def get_datasets_dir() -> Path:
    datasets_dir = get_data_dir().joinpath('datasets')
    return ensure_dir_exists(datasets_dir)


def get_data_dir() -> Path:
    data_dir = project_root().joinpath('data')
    return ensure_dir_exists(data_dir)


def ensure_dir_exists(dir_: Path) -> Path:
    if not dir_.exists():
        dir_.mkdir()
    return dir_


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent
