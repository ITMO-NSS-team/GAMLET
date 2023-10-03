from typing import Dict

import pandas as pd

from .dataset_meta_features_extractor import DatasetMetaFeaturesExtractor


# TODO: implement auto-detection of categorical (aka nominal) featires
# TODO: numeric features are non-categorical features (real-valued features?)
# TODO: allow to pass not only dataset CSV-file path but other options.
# TODO: currently for classification task only.
# TODO: is it okay to exlude target column from the datset?
class OpenMLDatasetMetaFeaturesExtractor(DatasetMetaFeaturesExtractor):
    """OpenML set of dataset meta-features extractor.

    Parameters
    ----------
    dataset: Path to the dataset CSV-file.
    features_preprocessors: Iterable of the dataset `FeaturesPreprocessor`. Default: `[]`.
    target_name: Name or index of target column in the dataset. Default: `"target"`.
    null_empty_string: Whether to consider an empty string as null value. Default: `True`.
    return_type: Type of return value. Can be `"dict"`, `"dataframe"` or `"ndarrray"`. Default: `"dict"`.
    """

    def _get_max_nominal_att_distinct_values(self) -> int:
        """Count maximal number of distinct values across all nominal attributes."""
        return -1  # TODO: implement

    def _get_majority_class_size(self) -> int:
        """Count the number of instances in the class with the largest number of instances."""
        return self.target.value_counts(dropna=True).max()

    def _get_minority_class_size(self) -> int:
        """Count the number of instances in the class with the least number of instances."""
        return self.target.value_counts(dropna=True).min()

    def _get_number_of_classes(self) -> int:
        """Count the number of classes in the target column."""
        return self.target.value_counts(dropna=True).shape[0]

    def _get_number_of_features(self) -> int:
        """Count the number of features in the dataset."""
        return self.dataset.shape[1]

    def _get_number_of_instances(self) -> int:
        """Count the number of instances in the dataset."""
        return self.dataset.shape[0]

    def _get_number_of_instances_with_missing_values(self) -> int:
        """Count the number of instances with missing values in the dataset."""
        return self.dataset.isnull().sum(axis=1)

    def _get_number_of_missing_values(self) -> int:
        """Count the number of missing values in all instances in the dataset."""
        return self.dataset.isnull().to_numpy().flatten().sum()

    def _get_number_of_numeric_features(self) -> int:
        """Count the number of numeric features."""
        # TODO: subject of changes
        return len(list(filter(pd.api.types.is_numeric_dtype, self.dataset.dtypes)))

    def _get_number_of_symbolic_features(self) -> int:
        """Count the number of symbolic features."""
        # TODO: subject of changes
        return len(list(filter(pd.api.types.is_string_dtype, self.dataset.dtypes)))

    def _get_features(self) -> Dict[str, int]:
        result = {}
        result["MajorityClassSize"] = self._get_majority_class_size()
        result["MinorityClassSize"] = self._get_minority_class_size()
        result["NumberOfClasses"] = self._get_number_of_classes()
        result["NumberOfFeatures"] = self._get_number_of_features()
        result["NumberOfInstances"] = self._get_number_of_instances()
        result["NumberOfInstancesWithMissingValues"] = self._get_number_of_instances_with_missing_values()
        result["NumberOfMissingValues"] = self._get_number_of_missing_values()
        result["NumberOfNumericFeatures"] = self._get_number_of_numeric_features()
        result["NumberOfSymbolicFeatures"] = self._get_number_of_symbolic_features()
        result["MaxNominalAttDistinctValues"] = self._get_max_nominal_att_distinct_values()
        return result
