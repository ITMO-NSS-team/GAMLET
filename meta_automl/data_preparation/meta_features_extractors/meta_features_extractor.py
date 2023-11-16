from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class MetaFeaturesExtractor(ABC):
    default_params: Optional[Dict[str, Any]] = None

    @abstractmethod
    def extract(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
