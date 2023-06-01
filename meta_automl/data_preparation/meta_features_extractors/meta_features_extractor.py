from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Iterable, Dict, Any, Type

import pandas as pd

from meta_automl.data_preparation.data_manager import DataManager


class MetaFeaturesExtractor:
    default_params: Optional[Dict[str, Any]] = None
    source_name: Optional[str] = None
    data_manager: Type[DataManager] = DataManager

    @abstractmethod
    def extract(self, datasets) -> pd.DataFrame:
        raise NotImplementedError()

    def _get_meta_features_cache(self, dataset_id: Any, meta_feature_names: Iterable[str]):
        cache = self.data_manager.get_meta_features_dict(str(dataset_id), self.source_name)
        if set(meta_feature_names) ^ cache.keys():
            return None
        else:
            return {mf_name: cache[mf_name] for mf_name in meta_feature_names}

    def _update_meta_features_cache(self, dataset_id: Any, meta_features_dict: Dict[str, Any]):
        self.data_manager.update_meta_features_dict(dataset_id, self.source_name, meta_features_dict)
