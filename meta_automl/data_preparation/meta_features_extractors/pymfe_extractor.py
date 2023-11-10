from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from tqdm import tqdm

from meta_automl.data_preparation.dataset import DatasetBase, DatasetIDType
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor

logger = logging.getLogger(__file__)


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

    def extract(self, datasets_or_ids: Sequence[Union[DatasetBase, DatasetIDType]],
                fill_input_nans: bool = False, use_cached: bool = True, update_cached: bool = True,
                fit_kwargs: Dict[str, Any] = None, extract_kwargs: Dict[str, Any] = None) -> pd.DataFrame:
        fit_kwargs = fit_kwargs or {}
        extract_kwargs = extract_kwargs or {}

        columns_or_rows = {}
        meta_feature_names = self._extractor.extract_metafeature_names()

        is_sum_none = True if "summary" in self.extractor_params and self.extractor_params["summary"] is None else False
        if is_sum_none:
            columns_or_rows["dataset"] = []
            columns_or_rows["feature"] = []
            columns_or_rows["value"] = []
            columns_or_rows["variable"] = []

        for dataset in tqdm(datasets_or_ids, desc='Extracting meta features of the datasets'):
            if isinstance(dataset, DatasetBase):
                dataset_id = dataset.id_
                dataset_class = dataset.__class__
            else:
                dataset_id = dataset
                dataset_class = self.datasets_loader.dataset_class
            logger.debug(
                f'{self.__class__.__name__}: extracting metafeatures of dataset {dataset_class.__name__}|{dataset_id}.'
            )
            meta_features_cached = self._get_meta_features_cache(dataset_id, dataset_class, meta_feature_names)

            if use_cached and meta_features_cached:
                columns_or_rows[dataset_id] = meta_features_cached
            else:
                if not isinstance(dataset, DatasetBase):
                    dataset = self._datasets_loader.load_single(dataset)
                dataset_data = dataset.get_data()
                x = dataset_data.x
                y = dataset_data.y
                cat_cols_indicator = dataset_data.categorical_indicator
                if fill_input_nans:
                    x = self.fill_nans(x, cat_cols_indicator)
                x = x.to_numpy()
                y = y.to_numpy()
                if cat_cols_indicator is not None:
                    cat_cols = [i for i, val in enumerate(cat_cols_indicator) if val]
                else:
                    cat_cols = 'auto'
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

                        columns_or_rows["dataset"].extend([dataset.id_] * dim_dataset)
                        columns_or_rows["variable"].extend(list(range(dim_dataset)))
                        columns_or_rows["feature"].extend([key] * dim_dataset)
                        columns_or_rows["value"].extend(value)
                else:
                    columns_or_rows[dataset.id_] = meta_features_extracted

        if is_sum_none:
            columns_or_rows = pd.DataFrame.from_dict(columns_or_rows)
        else:
            columns_or_rows = pd.DataFrame.from_dict(columns_or_rows, orient='index')
        return columns_or_rows

    @staticmethod
    def fill_nans(x: pd.DataFrame, cat_cols_indicator: Sequence[bool]):
        x_new = deepcopy(x)
        for idx, col in enumerate(x.columns):
            is_categorical = cat_cols_indicator[idx]
            if is_categorical:
                fill_value = x_new[col].mode(dropna=True)
            else:
                fill_value = x_new[col].median(skipna=True)
            x_new[col].fillna(fill_value, inplace=True)
        return x_new
