from typing import Dict, List

import numpy as np
import openml
import pandas as pd
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor


class OpenMLDatasetMetaFeaturesExtractor:
    """OpenML set of dataset meta-features extractor.

    Parameters
    ----------
    dataset_id : dataset id.
    meta_features_data_columns: List of names of meta-features to be extracted.
    return_type: Type of return value. Can be `"dict"`, `"dataframe"` or `"ndarrray"`. Default: `"dict"`.
    features_preprocessors: Wrapped features preprocessors.
    """

    meta_features_data_columns = [
        "MajorityClassSize",
        "MaxNominalAttDistinctValues",
        "MinorityClassSize",
        "NumberOfClasses",
        "NumberOfFeatures",
        "NumberOfInstances",
        "NumberOfInstancesWithMissingValues",
        "NumberOfMissingValues",
        "NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures"
    ]

    def __init__(
            self,
            dataset_id: int = None,
            meta_features_data_columns: List[str] = None,
            return_type: str = "dict",
            features_preprocessors: FeaturesPreprocessor = None,
        ):
        if meta_features_data_columns is not None:
            self.meta_features_data_columns = meta_features_data_columns
        self.return_type = return_type
        self.features_preprocessors = features_preprocessors

        if dataset_id is not None:
            self.meta_features = self._get_features(dataset_id)

    def _get_features(self, dataset_id: int) -> Dict[str, int]:
        if dataset_id is not None:
            dataset_info = openml.datasets.list_datasets([dataset_id,], output_format="dataframe").iloc[0].to_dict()
            meta_features = {}
            for key in self.meta_features_data_columns:
                try:
                    meta_features[key] = dataset_info[key]
                except KeyError:
                     meta_features[key] = -1
            return meta_features
        else:
            return self.meta_features

    def __call__(self, dataset_id: int = None) -> Dict[str, int]:
        features = self._get_features(dataset_id)
        if self.features_preprocessors is not None:
            features = self.features_preprocessors(features)

        if self.return_type == "dict":
            return features
        elif self.return_type == "dataframe":
            return pd.DataFrame.from_dict({k: [v,] for k, v in features})
        elif self.return_type == "ndarray":
            return np.array(list(features.values()))
