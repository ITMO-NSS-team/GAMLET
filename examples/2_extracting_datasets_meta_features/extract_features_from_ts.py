import os
from pathlib import Path

from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors.time_series_meta_features_extractor import \
    TimeSeriesFeaturesExtractor


def main():
    dataset_names = os.listdir(Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets'))[:10]

    loader = TimeSeriesDatasetsLoader()
    extractor = TimeSeriesFeaturesExtractor()

    datasets = loader.load(dataset_names)
    meta_features = extractor.extract(datasets)
    return meta_features

if __name__ == '__main__':
    result = main()
