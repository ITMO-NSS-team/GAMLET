import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors.time_series.time_series_meta_features_extractor import \
    TimeSeriesFeaturesExtractor
from meta_automl.meta_algorithm.dataset_similarity_assessors import KNeighborsSimilarityAssessor


def main():
    # Define datasets.
    dataset_names = os.listdir(Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets'))
    loader = TimeSeriesDatasetsLoader()
    datasets = loader.load(dataset_names)
    # Extract meta-features and load on demand.
    extractor = TimeSeriesFeaturesExtractor()
    meta_features = extractor.extract(datasets)
    # Preprocess meta-features, as KNN does not support NaNs.
    meta_features = meta_features.dropna(axis=1, how='any')
    # Split datasets to train (preprocessing) and test (actual meta-algorithm objects).
    mf_train, mf_test, did_train, did_test = train_test_split(meta_features, meta_features.index, train_size=0.75,
                                                              random_state=42)
    assessor = KNeighborsSimilarityAssessor(n_neighbors=3)
    assessor.fit(mf_train, did_train)
    # Get the closest datasets from train.
    return did_test, assessor.predict(mf_test, return_distance=True)


if __name__ == '__main__':
    result = main()
    print(result)
