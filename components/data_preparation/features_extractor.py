from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Iterable, Dict, Any, Union

import pandas as pd
from pymfe.mfe import MFE

from support.data_utils import get_meta_features_dict, update_meta_features_dict

from components.data_preparation.dataset import DatasetCache


class MetaFeaturesExtractor:
    DEFAULT_PARAMS = None
    SOURCE: Optional[str] = None

    def __init__(self, extractor_params=None):
        self.extractor_params = extractor_params or self.DEFAULT_PARAMS
        self._datasets_loader = None

    @abstractmethod
    def extract(self, datasets):
        raise NotImplementedError()

    def _get_meta_features_cache(self, dataset_name: str, meta_feature_names: Iterable[str]):
        cache = get_meta_features_dict(dataset_name, self.SOURCE)
        if set(meta_feature_names) ^ cache.keys():
            return None
        else:
            return {mf_name: cache[mf_name] for mf_name in meta_feature_names}

    def _update_meta_features_cache(self, dataset_name: str, meta_features_dict: Dict[str, Any]):
        update_meta_features_dict(dataset_name, self.SOURCE, meta_features_dict)


class PymfeExtractor(MetaFeaturesExtractor):
    DEFAULT_PARAMS = {'groups': 'default'}
    SOURCE = 'pymfe'

    def __init__(self, extractor_params=None):
        super().__init__(extractor_params)
        self.extractor = MFE(**self.extractor_params)

    def extract(self, datasets: List[Union[DatasetCache, str]]):
        meta_features = {}
        meta_feature_names = self.extractor.extract_metafeature_names()
        for dataset in datasets:
            if isinstance(dataset, str):
                dataset = DatasetCache(dataset)
            if mfs := self._get_meta_features_cache(dataset.name, meta_feature_names):
                meta_features[dataset.name] = mfs
            else:
                loaded_dataset = dataset.load()
                cat_cols = [i for i, val in enumerate(loaded_dataset.categorical_indicator) if val]
                mfe = self.extractor.fit(loaded_dataset.X, loaded_dataset.y, cat_cols=cat_cols)
                feature_names, dataset_features = mfe.extract(out_type=tuple)
                mfs = dict(zip(feature_names, dataset_features))
                self._update_meta_features_cache(dataset.name, mfs)
                meta_features[dataset.name] = mfs
        meta_features = pd.DataFrame.from_dict(meta_features, orient='index')
        return meta_features
