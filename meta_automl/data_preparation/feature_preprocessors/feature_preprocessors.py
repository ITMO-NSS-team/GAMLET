import os
import pickle
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def explode_ungrouped_metafeatures(df: pd.DataFrame, n_features: int) -> pd.DataFrame:
    output = []
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
    return pd.concat(output)


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
            extractor_params: Dict[str, Any] = None,
    ):
        if load_path is not None:
            print("Load from file. `preprocessors` argument will be ignored.")
            with open(load_path, 'rb') as f:
                self.preprocessors = pickle.load(f)
        elif preprocessors is not None:
            self.preprocessors = preprocessors
        else:
            self.preprocessors = dict()
        self.extractor_params = extractor_params
        conditions = [
            self.extractor_params is not None,
            "summary" in self.extractor_params,
            self.extractor_params["summary"] is None
        ]
        self.is_sum_none = all(conditions)
        if self.is_sum_none:
            self.features = self.extractor_params["features"]

    def __call__(
            self,
            data: Dict[Union[str, int], Union[int, float]],
            single: bool = True,
    ) -> Dict[str, Union[float, np.ndarray]]:
        return self.transform(data, single)

    def transform(
            self,
            data: Dict[Union[str, int], Union[int, float]],
            single: bool = False,
    ) -> Dict[str, Union[float, np.ndarray]]:
        result = data.copy()

        if self.is_sum_none:
            result = explode_ungrouped_metafeatures(result, len(self.features))
            for key in self.features:
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

    def fit(self, data: pd.DataFrame, save_path: Optional[str] = None):
        if self.is_sum_none:
            for key in self.features:
                if key not in self.preprocessors:
                    self.preprocessors[key] = StandardScaler()
                self.preprocessors[key].fit(data.loc[(data["feature"] == key), "value"].values.reshape(-1, 1))
        else:
            for key in data.columns:
                if key not in self.preprocessors:
                    self.preprocessors[key] = StandardScaler()
                self.preprocessors[key].fit(data[key].values.reshape(-1, 1))
        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(self.preprocessors, f)
