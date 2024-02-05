import os
from pathlib import Path

from gamlet.components.datasets_loaders import TimeSeriesDatasetsLoader
from gamlet.components.meta_features_extractors import TimeSeriesFeaturesExtractor
from gamlet.data_preparation.file_system import get_project_root


def main():
    dataset_names = os.listdir(Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets'))[:10]

    loader = TimeSeriesDatasetsLoader()
    extractor = TimeSeriesFeaturesExtractor()

    datasets = loader.load(dataset_names)
    meta_features = extractor.extract(datasets)
    return meta_features


if __name__ == '__main__':
    result = main()
    print(result)
