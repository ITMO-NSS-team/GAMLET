import pickle
from typing import Any, Dict, Optional, Union

import numpy as np


class FeaturesPreprocessor:
    """Wrapper over features preporcessors.

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
                preprocessors = pickle.load(f)
        else:
            self.preprocessors = preprocessors

    def __call__(
        self,
        data: Dict[Union[str, int], Union[int, float]],
        single: bool = True,
    ) -> Dict[str, Union[float, np.ndarray]]:
        return self.transform(data, single)

    def transform(
        self,
        data: Dict[Union[str, int], Union[int, float]],
        single: bool = True,
    ) -> Dict[str, Union[float, np.ndarray]]:
        result = {}
        for key, value in data.items():
            if single:
                result[key] = self.preprocessors[key].transform(np.array(value).reshape(1, 1)).item(0)
            else:
                result[key] = self.preprocessors[key].transform(np.array(value).reshape(-1, 1))
        return result

    def fit(self, data: Dict[str, Union[int, float]], save_path: Optional[str] = None):
        for key, value in data.items():
            self.preprocessors[key].fit(np.array(value).reshape(-1, 1))
        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(self.preprocessors, f)
