from __future__ import annotations

from typing import List, Union

import pandas as pd
from pymfe.mfe import MFE

from meta_automl.data_preparation.dataset import DatasetCache, NoCacheError
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor


class PymfeExtractor(MetaFeaturesExtractor):
    DEFAULT_PARAMS = {'groups': 'default'}
    SOURCE = 'pymfe'

    def __init__(self):
        self.extractor_params = self.DEFAULT_PARAMS
        self._datasets_loader = None
        self._extractor = None

    def fit(self, extractor_params=None, datasets_loader=None) -> PymfeExtractor:
        self._datasets_loader = datasets_loader
        self.extractor_params = extractor_params if extractor_params is not None else self.extractor_params
        self._extractor = MFE(**self.extractor_params)
        return self

    @property
    def datasets_loader(self):
        if not self._datasets_loader:
            raise ValueError("Datasets loader not provided!")
        return self._datasets_loader

    def extract(self, datasets: List[Union[DatasetCache, str]]) -> pd.DataFrame:
        meta_features = {}
        meta_feature_names = self._extractor.extract_metafeature_names()
        for dataset in datasets:
            if isinstance(dataset, str):
                dataset = DatasetCache(dataset)
            if mfs := self._get_meta_features_cache(dataset.name, meta_feature_names):
                meta_features[dataset.name] = mfs
            else:
                try:
                    loaded_dataset = dataset.load()
                except NoCacheError:
                    loaded_dataset = self.datasets_loader.load_single(dataset.name).load()
                cat_cols = [i for i, val in enumerate(loaded_dataset.categorical_indicator) if val]
                mfe = self._extractor.fit(loaded_dataset.X, loaded_dataset.y, cat_cols=cat_cols)
                feature_names, dataset_features = mfe.extract(out_type=tuple)
                mfs = dict(zip(feature_names, dataset_features))
                self._update_meta_features_cache(dataset.name, mfs)
                meta_features[dataset.name] = mfs
        meta_features = pd.DataFrame.from_dict(meta_features, orient='index')
        return meta_features
