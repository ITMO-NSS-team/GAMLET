from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Iterable, Dict, Any

import pandas as pd

from meta_automl.data_preparation.data_directory_manager import DataDirectoryManager


class MetaFeaturesExtractor:
    DEFAULT_PARAMS: Optional[Dict[str, Any]] = None
    SOURCE: Optional[str] = None

    @abstractmethod
    def fit(self, *args, **kwargs) -> MetaFeaturesExtractor:
        raise NotImplementedError()

    @abstractmethod
    def extract(self, datasets) -> pd.DataFrame:
        raise NotImplementedError()

    def _get_meta_features_cache(self, dataset_name: str, meta_feature_names: Iterable[str]):
        cache = DataDirectoryManager.get_meta_features_dict(dataset_name, self.SOURCE)
        if set(meta_feature_names) ^ cache.keys():
            return None
        else:
            return {mf_name: cache[mf_name] for mf_name in meta_feature_names}

    def _update_meta_features_cache(self, dataset_name: str, meta_features_dict: Dict[str, Any]):
        DataDirectoryManager.update_meta_features_dict(dataset_name, self.SOURCE, meta_features_dict)
