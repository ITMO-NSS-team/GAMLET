from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Iterable, Dict, Any, Union

import openml
import pandas as pd
from pymfe.mfe import MFE

from support.data_utils import get_meta_features_dict, update_meta_features_dict

from components.data_preparation.dataset import DatasetCache, NoCacheError


class MetaFeaturesExtractor:
    DEFAULT_PARAMS = None
    SOURCE: Optional[str] = None

    @abstractmethod
    def fit(self, *args, **kwargs) -> MetaFeaturesExtractor:
        raise NotImplementedError()

    @abstractmethod
    def extract(self, datasets) -> pd.DataFrame:
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


class OpenMLFeaturesExtractor(MetaFeaturesExtractor):  # Demo
    DEFAULT_PARAMS = {}
    SOURCE = 'openml_mf'

    def fit(self) -> OpenMLFeaturesExtractor:
        return self

    def extract(self, datasets: List[Union[DatasetCache, str]]) -> pd.DataFrame:
        datasets = [d if isinstance(d, str) else d.name for d in datasets]
        datasets = [openml.datasets.get_dataset(d).dataset_id for d in datasets]
        df_datasets = openml.datasets.list_datasets(data_id=datasets, output_format='dataframe')
        df_tasks = openml.tasks.list_tasks(output_format="dataframe").drop_duplicates()
        df_datasets = df_datasets.rename(columns={'did': 'data_id', 'name': 'data_name'})
        df_tasks = df_tasks.rename(columns={'did': 'data_id', 'tid': 'task_id', 'name': 'data_name'})

        df = pd.merge(df_datasets,
                      df_tasks[list(set(df_tasks.columns) - set(df_datasets.columns)) + ['task_id', 'data_id']],
                      on=['data_id'])

        columns = set(df.columns) - {'Unnamed: 0', 'run_id', 'setup_id', 'flow_id', 'flow_name', 'data_id', 'data_name',
                                     'upload_time', 'uploader', 'uploader_name', 'values', 'array_data', 'version',
                                     'status', 'target_value', 'target_feature_right', 'target_feature_event',
                                     'quality_measure', 'source_data', 'source_data_labeled', 'number_samples',
                                     'external_version'}

        df = df[columns]
        return df
