from typing import Any, Dict, Union

import numpy as np


class FeaturesPreprocessor:
    """Wrapper over features preporcessors.

    Parameters:
    -----------
    preprocessors: Dict of features preprocessor. Preprocessors should implement `transform` method.

    """
    def __init__(self, preprocessors: Dict[Union[str, int], Any]):
        self.preprocessors = preprocessors

    def __call__(self, data: Dict[Union[str, int], Union[int, float]]):
        result = {}
        for key, value in data.items():
            result[key] = self.preprocessors[key].transform(np.ndarray(value).reshape(1, 1)).item(0)
        return result