import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors.time_series_meta_features_extractor import \
    TimeSeriesFeaturesExtractor


def main():
    # Define datasets.
    dataset_names = os.listdir(Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets'))

    loader = TimeSeriesDatasetsLoader()
    datasets = loader.load(dataset_names)
    # Extract meta-features and load on demand.
    extractor = TimeSeriesFeaturesExtractor()
    #meta_features = extractor.extract(datasets)
    meta_features = pd.read_csv('../../data/knowledge_base_time_series_0/meta_features_ts.csv')
    # Preprocess meta-features, as KNN does not support NaNs.
    meta_features = meta_features.dropna(axis=1, how='any')
    #meta_features.to_csv('meta_features_ts.csv', index=False)
    # Select the KMeans clustering algorithm
    pca = PCA(n_components=2)
    X = pca.fit_transform(meta_features.values)
    X = StandardScaler().fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1])
    # res = []
    # variants = list(range(2, 100))
    #
    #
    # kmeans = DBSCAN(eps=0.5, min_samples=10)
    #
    # # Apply the clustering algorithm to your data
    # kmeans_labels = kmeans.fit_predict(X)
    # print(len(np.unique(kmeans_labels)))
    #
    # # Calculate the silhouette score
    # score = silhouette_score(X, kmeans_labels)
    # res.append(score)
    # print(f'Silhouette score for  clusters: {score:.2f}')
    # # plt.plot(variants, res)
    plt.show()


if __name__ == '__main__':
    main()
