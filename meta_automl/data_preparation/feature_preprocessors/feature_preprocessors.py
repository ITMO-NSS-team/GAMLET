import pickle
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FeaturesPreprocessor:
    """Wrapper over features preporcessors.

    If a feature preprocessor is not defined, use `sklearn.preprocessing.StandartScaler` by default during fit.

    Parameters:
    -----------
    preprocessors: Dict of features preprocessor. Preprocessors should implement `fit` and `transform` methods.
                   Can be ommited if `load` argumnet is specified.
    load_path: Path to load preprocessors from. If specified, `preprocessors` argument will be ignored.
    """

    def __init__(self, preprocessors: Dict[Union[str, int], Any] = None, load_path: str = None):
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
        for key in data.columns:
            result[key] = self.preprocessors[key].transform(data[key].values.reshape(-1, 1))
        # for key, value in data.items():
        #     if single:
        #         result[key] = self.preprocessors[key].transform(np.array(value).reshape(1, 1)).item(0)
        #     else:
        #         result[key] = self.preprocessors[key].transform(np.array(value).reshape(-1, 1))
        return result

    def fit(self, data: Dict[str, Union[int, float]], save_path: Optional[str] = None):
        for key in data.columns:
            if key not in self.preprocessors:
                self.preprocessors[key] = StandardScaler()
            self.preprocessors[key].fit(data[key].values.reshape(-1, 1))
        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(self.preprocessors, f)
