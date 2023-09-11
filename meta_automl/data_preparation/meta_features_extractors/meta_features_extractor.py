from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Optional, Iterable, Dict, Any

import pandas as pd

from meta_automl.data_preparation.dataset import DatasetIDType, DatasetBase
from meta_automl.data_preparation.file_system import (CacheOperator, get_local_meta_features,
                                                      update_local_meta_features)


class MetaFeaturesExtractor(ABC, CacheOperator):
    default_params: Optional[Dict[str, Any]] = None

    @abstractmethod
    def extract(self, datasets) -> pd.DataFrame:
        raise NotImplementedError()

    def _get_meta_features_cache(self, dataset: DatasetBase, meta_feature_names: Iterable[str]):
        dataset_class = dataset.__class__
        dataset_id = str(dataset.id_)
        cache = get_local_meta_features(self.__class__, dataset_class, dataset_id)
        if set(meta_feature_names) ^ cache.keys():
            return None
        else:
            return {mf_name: cache[mf_name] for mf_name in meta_feature_names}

    def _update_meta_features_cache(self, dataset: DatasetBase, meta_features_dict: Dict[str, Any]):
        dataset_class = dataset.__class__
        dataset_id = str(dataset.id_)
        update_local_meta_features(self.__class__, dataset_class, dataset_id, meta_features_dict)
