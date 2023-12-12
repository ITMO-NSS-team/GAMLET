from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from meta_automl.data_preparation.meta_features_extractors.dataset_meta_features import DatasetMetaFeatures


class MetaFeaturesExtractor(ABC):
    default_params: Optional[Dict[str, Any]] = None

    @abstractmethod
    def extract(self, *args, **kwargs) -> DatasetMetaFeatures:
        raise NotImplementedError()
