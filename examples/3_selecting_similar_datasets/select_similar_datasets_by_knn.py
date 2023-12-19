from sklearn.model_selection import train_test_split

from gamlet.components.datasets_loaders import OpenMLDatasetsLoader
from gamlet.components.meta_features_extractors import PymfeExtractor
from gamlet.components.dataset_similarity_assessors import KNeighborsSimilarityAssessor


def main():
    # Define datasets.
    dataset_names = ['monks-problems-1', 'apsfailure', 'australian', 'bank-marketing']
    datasets = OpenMLDatasetsLoader().load(dataset_names, allow_names=True)
    # Extract meta-features and load on demand.
    extractor = PymfeExtractor(groups='general')
    meta_features = extractor.extract(datasets)
    # Preprocess meta-features, as KNN does not support NaNs.
    meta_features = meta_features.dropna(axis=1, how='any')
    # Split datasets to train (preprocessing) and test (actual meta-algorithm objects).
    x_train, x_test = train_test_split(meta_features, train_size=0.75, random_state=42)
    y_train = x_train.index
    assessor = KNeighborsSimilarityAssessor(n_neighbors=3)
    assessor.fit(x_train, y_train)
    # Get models for the best fitting datasets from train.
    return x_test.index, assessor.predict(x_test, return_distance=True)


if __name__ == '__main__':
    result = main()
