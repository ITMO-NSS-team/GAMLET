import os
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import QuantileTransformer

from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors.time_series.time_series_meta_features_extractor import \
    TimeSeriesFeaturesExtractor

p = Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets')
len_d = {i: len(pd.read_csv(Path(p, i, 'data.csv'))) for i in os.listdir(p)}

tresholds = np.array([100, 200, 300, 500, 700, 1000, 2000, 4000])

len_d_classes = {k: tresholds[np.argmin(np.abs(tresholds - v))] for k, v in len_d.items()}


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
    idx = meta_features.index.values
    # Dimension reduction
    dim_reduction = umap.UMAP(n_components=2, min_dist=0.5, n_neighbors=10)
    X = QuantileTransformer(output_distribution='normal').fit_transform(meta_features)
    X = dim_reduction.fit_transform(X)

    plt.title('UMAP projection of the data')
    colors = {'D': 'blue', 'W': 'orange', 'M': 'red', 'Q': 'black', 'Y': 'purple'}

    for item in colors:
        label = item[0]
        plt.plot([0], [0], 'o', c=colors[item], markersize=3, label=str(label))
    plt.legend()
    plt.scatter(X[:, 0], X[:, 1], c=[colors[i[3]] for i in idx],
                s=[len_d[i] / 50 if i in len_d else 10 for i in idx])
    plt.grid()
    plt.show()

    cluster_algo = DBSCAN(eps=0.7)

    # Apply the clustering algorithm to your data
    cluster_labels = cluster_algo.fit_predict(X)
    df_labels = pd.Series(index=idx, data=cluster_labels)
    df_labels.to_csv('labels.csv')


if __name__ == '__main__':
    main()
