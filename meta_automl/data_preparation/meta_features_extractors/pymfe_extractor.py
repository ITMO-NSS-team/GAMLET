from __future__ import annotations

from typing import List, Union, Dict, Any

import pandas as pd
from pymfe.mfe import MFE

from meta_automl.data_preparation.dataset import DatasetCache
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor


class PymfeExtractor(MetaFeaturesExtractor):
    DEFAULT_PARAMS = {'groups': 'default'}
    SOURCE = 'pymfe'

    def __init__(self, extractor_params: Dict[str, Any] = None, datasets_loader: DatasetsLoader = None):
        self.extractor_params = extractor_params if extractor_params is not None else self.DEFAULT_PARAMS
        self._datasets_loader = datasets_loader or OpenMLDatasetsLoader()
        self._extractor = MFE(**self.extractor_params)

    @property
    def datasets_loader(self) -> DatasetsLoader:
        if not self._datasets_loader:
            raise ValueError("Datasets loader not provided!")
        return self._datasets_loader

    def extract(self, datasets: List[Union[DatasetCache, str]], fill_nans: bool = False) -> pd.DataFrame:
        meta_features = {}
        meta_feature_names = self._extractor.extract_metafeature_names()
        load_dataset = self.datasets_loader.cache_to_memory
        for dataset in datasets:
            if isinstance(dataset, str):
                dataset = DatasetCache(dataset)
            if mfs := self._get_meta_features_cache(dataset.name, meta_feature_names):
                meta_features[dataset.name] = mfs
            else:
                loaded_dataset = load_dataset(dataset)
                cat_cols = [i for i, val in enumerate(loaded_dataset.categorical_indicator) if val]
                x = loaded_dataset.x
                y = loaded_dataset.y
                if fill_nans:
                    x = self.fill_nans(x)
                mfe = self._extractor.fit(x, y, cat_cols=cat_cols)
                feature_names, dataset_features = mfe.extract(out_type=tuple)
                mfs = dict(zip(feature_names, dataset_features))
                self._update_meta_features_cache(dataset.name, mfs)
                meta_features[dataset.name] = mfs
        meta_features = pd.DataFrame.from_dict(meta_features, orient='index')
        return meta_features

    @staticmethod
    def fill_nans(x):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.fillna(x.median())
        return x.to_numpy()
