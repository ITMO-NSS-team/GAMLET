from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import List, Union, Dict, Any

import pandas as pd
import numpy as np
from pymfe.mfe import MFE

from meta_automl.data_preparation.dataset import DatasetBase, DatasetIDType
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor
import json


class PymfeExtractor(MetaFeaturesExtractor):
    default_params = {'groups': 'default'}

    def __init__(self, extractor_params: Dict[str, Any] = None, datasets_loader: DatasetsLoader = None):
        self.extractor_params = extractor_params if extractor_params is not None else self.default_params
        self._datasets_loader = datasets_loader or OpenMLDatasetsLoader()
        self._extractor = MFE(**self.extractor_params)

    @property
    def datasets_loader(self) -> DatasetsLoader:
        if not self._datasets_loader:
            raise ValueError("Datasets loader not provided!")
        return self._datasets_loader

    def extract(self, datasets_or_ids: List[Union[DatasetBase, DatasetIDType]],
                fill_input_nans: bool = False, use_cached: bool = True, update_cached: bool = True,
                fit_kwargs: Dict[str, Any] = None, extract_kwargs: Dict[str, Any] = None) -> pd.DataFrame:
        fit_kwargs = fit_kwargs or {}
        extract_kwargs = extract_kwargs or {}

        meta_features_dict = {}
        meta_feature_names = self._extractor.extract_metafeature_names()

        is_sum_none = True if "summary" in self.extractor_params and self.extractor_params["summary"] is None else False
        if is_sum_none:
            meta_features_dict["dataset"] = []
            meta_features_dict["feature"] = []
            meta_features_dict["value"] = []
            meta_features_dict["variable"] = []

        for dataset in datasets_or_ids:
            if isinstance(dataset, DatasetBase):
                dataset_id = dataset.id_
                dataset_class = dataset.__class__
            else:
                dataset_id = dataset
                dataset_class = self.datasets_loader.dataset_class
            meta_features_cached = self._get_meta_features_cache(dataset_id, dataset_class, meta_feature_names)

            logging.critical(f'Extracting meta features of the dataset {dataset}...')
            if use_cached and meta_features_cached:
                meta_features_dict[dataset_id] = meta_features_cached
            else:
                if not isinstance(dataset, DatasetBase):
                    dataset = self._datasets_loader.load_single(dataset)
                dataset_data = dataset.get_data()
                cat_cols_indicator = dataset_data.categorical_indicator
                if cat_cols_indicator is not None:
                    cat_cols = [i for i, val in enumerate(cat_cols_indicator) if val]
                else:
                    cat_cols = 'auto'
                x = dataset_data.x.to_numpy()
                y = dataset_data.y.to_numpy()
                if fill_input_nans:
                    x = self.fill_nans(x)
                fit_extractor = self._extractor.fit
                fit_extractor = partial(fit_extractor, x, y, cat_cols=cat_cols, **fit_kwargs)
                try:
                    mfe = fit_extractor(transform_cat=None)
                except RecursionError:
                    warnings.warn('PyMFE did not manage to do fit. Trying "one-hot" categorical encoder...')
                    mfe = fit_extractor(transform_cat='one-hot')
                feature_names, dataset_features = mfe.extract(out_type=tuple, **extract_kwargs)
                meta_features_extracted = dict(zip(feature_names, dataset_features))

                if update_cached:
                    self._update_meta_features_cache(dataset_id, dataset_class, meta_features_extracted)
                if is_sum_none:
                    dim_dataset = x.shape[1]

                    for key, value in meta_features_extracted.items():
                        value = value.tolist() if (isinstance(value, np.ndarray)) else [value] * dim_dataset
                        if len(value) == 0 or len(value) > dim_dataset:
                            value = [np.nan] * dim_dataset
                        if len(value) < dim_dataset:
                            value = [value[0]] * dim_dataset

                        meta_features_dict["dataset"].extend([dataset.id_] * dim_dataset)
                        meta_features_dict["variable"].extend(list(range(dim_dataset)))
                        meta_features_dict["feature"].extend([key] * dim_dataset)
                        meta_features_dict["value"].extend(value)
                else:
                    meta_features_dict[dataset.id_] = meta_features_extracted

        if is_sum_none:
            meta_features_dict = pd.DataFrame.from_dict(meta_features_dict)
        else:
            meta_features_dict = pd.DataFrame.from_dict(meta_features_dict, orient='index')
        return meta_features_dict

    @staticmethod
    def fill_nans(x):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.fillna(x.median())
        return x.to_numpy()
