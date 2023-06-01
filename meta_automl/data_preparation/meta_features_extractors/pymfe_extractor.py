from __future__ import annotations

from typing import List, Union, Dict, Any

import pandas as pd
from golem.core.log import default_log
from pymfe.mfe import MFE

from meta_automl.data_preparation.dataset import DatasetBase, DatasetIDType
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor


class PymfeExtractor(MetaFeaturesExtractor):
    default_params = {'groups': 'default'}
    source_name = 'pymfe'

    def __init__(self, extractor_params: Dict[str, Any] = None, datasets_loader: DatasetsLoader = None):
        self.extractor_params = extractor_params if extractor_params is not None else self.default_params
        self._datasets_loader = datasets_loader or OpenMLDatasetsLoader()
        self._extractor = MFE(**self.extractor_params)
        self._logger = default_log(self)

    @property
    def datasets_loader(self) -> DatasetsLoader:
        if not self._datasets_loader:
            raise ValueError("Datasets loader not provided!")
        return self._datasets_loader

    def extract(self, datasets_or_ids: List[Union[DatasetBase, Union[DatasetBase, DatasetIDType]]],
                fill_input_nans: bool = False, use_cached: bool = True, update_cached: bool = True) -> pd.DataFrame:
        meta_features = {}
        meta_feature_names = self._extractor.extract_metafeature_names()

        for dataset in datasets_or_ids:
            if not isinstance(dataset, DatasetBase):
                dataset = self._datasets_loader.load_single(dataset)

            self._logger.info(f'Extracting meta features of the dataset {dataset}...')
            if (use_cached and
                    (mfs := self._get_meta_features_cache(dataset.id_, meta_feature_names))):
                meta_features[dataset.id_] = mfs
            else:
                loaded_dataset = dataset.get_data(dataset_format='array')
                cat_cols = [i for i, val in enumerate(loaded_dataset.categorical_indicator) if val]
                x = loaded_dataset.x
                y = loaded_dataset.y
                if fill_input_nans:
                    x = self.fill_nans(x)
                mfe = self._extractor.fit(x, y, cat_cols=cat_cols)
                feature_names, dataset_features = mfe.extract(out_type=tuple)
                mfs = dict(zip(feature_names, dataset_features))
                if update_cached:
                    self._update_meta_features_cache(dataset.id_, mfs)
                meta_features[dataset.id_] = mfs
        meta_features = pd.DataFrame.from_dict(meta_features, orient='index')
        return meta_features

    @staticmethod
    def fill_nans(x):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.fillna(x.median())
        return x.to_numpy()
