from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from gamlet.components.meta_features_extractors import DatasetMetaFeatures


class MetaFeaturesExtractor(ABC):
    default_params: Optional[Dict[str, Any]] = None

    @abstractmethod
    def extract(self, *args, **kwargs) -> DatasetMetaFeatures:
        raise NotImplementedError()
