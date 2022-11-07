from pathlib import Path

import openml
import pandas as pd
from scipy.io.arff import loadarff


def get_openml_dataset(dataset_id) -> Path:
    openml_dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
    name = openml_dataset.name
    dataset_file = get_datasets_dir().joinpath(name).with_suffix('.csv')
    if not dataset_file.exists():
        raw_file = openml.datasets.get_dataset(dataset_id, download_data=True).data_file
        arff_to_csv(raw_file, dataset_file)
    return dataset_file


def arff_to_csv(input_path, output_path):
    raw_data = loadarff(input_path)
    df = pd.DataFrame(raw_data[0])
    df.to_csv(output_path)


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
