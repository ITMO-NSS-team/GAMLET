from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor


class DatasetMetaFeaturesExtractor:
    """Base dataset meta-feature extractor class.

    One should inherit this class to imlement own extractor.

    Parameters
    ----------
    dataset: Path to the dataset CSV-file.
    features_preprocessors: Iterable of the dataset `FeaturesPreprocessor`. Default: `[]`.
    target_name: Name or index of target column in the dataset. Default: `"target"`.
    null_empty_string: Whether to consider an empty string as null value. Default: `True`.
    return_type: Type of return value. Can be `"dict"`, `"dataframe"` or `"ndarrray"`. Default: `"dict"`.
    """

    def __init__(
        self,
        dataset: str,
        features_preprocessors: Optional[Iterable[FeaturesPreprocessor]] = [],
        target_name: Optional[Union[str, int]] = "target",
        null_empty_string: Optional[bool] = True,
        return_type: Optional[str] = "dict"
    ):
        possible_return_types = ["dict", "dataframe", "ndarray"]
        assert_message = f"Expected `return_type` is of {possible_return_types}, got {return_type}"
        assert return_type in possible_return_types, assert_message

        self.features_preprocessors = features_preprocessors
        self.return_type = return_type
        self.dataset: pd.DataFrame = pd.read_csv(dataset)
        if null_empty_string:
            self.dataset.replace("", np.nan, inplace=True)
        self.target = self.dataset.pop(target_name)

    def _get_features(self) -> Dict[str, int]:
        raise NotImplementedError("One should override this method in inherited class.")

    def __call__(self) -> Dict[str, int]:
        features = self._get_features()
        for preprocessor in self.features_preprocessors:
            features = preprocessor(features)
        if self.return_type == "dict":
            return features
        elif self.return_type == "dataframe":
            return pd.DataFrame.from_dict({k: [v,] for k, v in features})
        elif self.return_type == "ndarray":
            return np.array(list(features.values()))