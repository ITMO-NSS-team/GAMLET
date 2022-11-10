from __future__ import annotations

from abc import abstractmethod
from typing import List, TYPE_CHECKING

import pandas as pd
from pymfe.mfe import MFE

if TYPE_CHECKING:
    from components.dataset import Dataset


class MetaFeaturesExtractor:
    DEFAULT_FEATURES = None

    def __init__(self, features=None):
        self.features = features or self.DEFAULT_FEATURES

    def __call__(self, datasets):
        return self._extract_features(datasets)

    @abstractmethod
    def _extract_features(self, datasets):
        raise NotImplementedError()


class PymfeExtractor(MetaFeaturesExtractor):
    DEFAULT_FEATURES = 'default'

    def _extract_features(self, datasets: List[Dataset]):
        meta_features = {}
        for dataset in datasets:
            mfe = MFE(self.features).fit(dataset.X.to_numpy(), dataset.y.to_numpy())
            feature_names, dataset_features = mfe.extract(out_type=tuple)
            meta_features[dataset.name] = dict(zip(feature_names, dataset_features))
        meta_features = pd.DataFrame.from_dict(meta_features, orient='index')
        return meta_features
