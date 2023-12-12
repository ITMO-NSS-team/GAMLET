import os
import pickle
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from meta_automl.data_preparation.meta_features_extractors.dataset_meta_features import DatasetMetaFeatures


def explode_ungrouped_metafeatures(df: DatasetMetaFeatures) -> DatasetMetaFeatures:
    output = []
    n_features = len(df.features)
    for index, row in df.iterrows():
        df_row = pd.DataFrame(row)
        df_row['value'] = df_row['value'].apply(
            (lambda value: value if isinstance(value, np.ndarray) else [value] * n_features))
        df_row = df_row.drop(
            df_row[(df_row['value'].apply(len) > n_features) | (df_row['value'].apply(len) == 0)].index)
        df_row['value'] = df_row['value'].apply(
            (lambda value: value if len(value) == n_features else [value[0]] * n_features))
        df_row = df_row.explode('value')
        df_row['feature'] = df_row.groupby('variable').cumcount()
        df_row['dataset'] = index
        df_row = df_row[['dataset', 'variable', 'feature', 'value']]
        output.append(df_row)
    return DatasetMetaFeatures(pd.concat(output), is_summarized=True, features=df.features)


class FeaturesPreprocessor:
    """Wrapper over features preprocessors.

    If a feature preprocessor is not defined, use `sklearn.preprocessing.StandartScaler` by default during fit.

    Parameters:
    -----------
    preprocessors: Dict of features preprocessor. Preprocessors should implement `fit` and `transform` methods.
                   Can be omitted if `load` argument is specified.
    load_path: Path to load preprocessors from. If specified, `preprocessors` argument will be ignored.
    """

    def __init__(
            self,
            preprocessors: Dict[Union[str, int], Any] = None,
            load_path: os.PathLike = None,
    ):
        if load_path is not None:
            print("Load from file. `preprocessors` argument will be ignored.")
            with open(load_path, 'rb') as f:
                self.preprocessors = pickle.load(f)
        elif preprocessors is not None:
            self.preprocessors = preprocessors
        else:
            self.preprocessors = dict()

    def __call__(
            self,
            data: DatasetMetaFeatures,
            single: bool = True,
    ) -> DatasetMetaFeatures:
        return self.transform(data, single)

    def transform(
            self,
            data: DatasetMetaFeatures,
            single: bool = False,
    ) -> DatasetMetaFeatures:
        result = data.copy()

        if data.is_summarized:
            result = explode_ungrouped_metafeatures(result)
            for key in data.features:
                data_ = data.loc[(data["feature"] == key), "value"].values.reshape(-1, 1)
                result.loc[(data["feature"] == key), "value"] = self.preprocessors[key].transform(data_)
        else:
            for key in data.columns:
                result[key] = self.preprocessors[key].transform(data[key].values.reshape(-1, 1))
        # for key, value in data.items():
        #     if single:
        #         result[key] = self.preprocessors[key].transform(np.array(value).reshape(1, 1)).item(0)
        #     else:
        #         result[key] = self.preprocessors[key].transform(np.array(value).reshape(-1, 1))
        return result

    def fit(self, data: DatasetMetaFeatures, save_path: Optional[str] = None):
        if data.is_summarized:
            for key in data.columns:
                if key not in self.preprocessors:
                    self.preprocessors[key] = StandardScaler()
                self.preprocessors[key].fit(data[key].values.reshape(-1, 1))
        else:
            for key in data.features:
                if key not in self.preprocessors:
                    self.preprocessors[key] = StandardScaler()
                self.preprocessors[key].fit(data.loc[(data["feature"] == key), "value"].values.reshape(-1, 1))

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(self.preprocessors, f)
