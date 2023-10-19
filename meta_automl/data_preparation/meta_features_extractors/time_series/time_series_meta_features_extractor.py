import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from tqdm.auto import tqdm

from meta_automl.data_preparation.dataset import DatasetBase, DatasetIDType
from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor

TS_FEATURES_COUNT = 184


class TimeSeriesFeaturesExtractor(MetaFeaturesExtractor):
    default_params: Optional[Dict[str, Any]] = None

    def __init__(self, custom_path=None):
        self._extractor = self._load_extractor()
        self._datasets_loader = TimeSeriesDatasetsLoader(custom_path=custom_path)
        self.meta_feature_names = np.arange(TS_FEATURES_COUNT)

    def extract(self,
                datasets_or_ids: List[Union[DatasetBase, DatasetIDType]],
                fill_input_nans: bool = False, use_cached: bool = True, update_cached: bool = True,
                fit_kwargs: Dict[str, Any] = None, extract_kwargs: Dict[str, Any] = None) -> pd.DataFrame:

        meta_features = {}
        logging.critical('Extracting meta features of the time series datasets...')

        with IndustrialModels():
            for d in tqdm(datasets_or_ids, desc='Feature_generation'):
                if not isinstance(d, DatasetBase):
                    d = self._datasets_loader.load_single(d)
                idx = d.id_
                if (use_cached and
                        (mfs := self._get_meta_features_cache(d, meta_feature_names))):
                    meta_features[idx] = mfs.values()
                else:

                    features = d.get_data().x

                    input_data = InputData(idx=np.array([0]), features=np.array(features).reshape(1, -1), target=None,
                                           task=Task(TaskTypesEnum.classification),
                                           data_type=DataTypesEnum.table)
                    pred = self._extractor.root_node.predict(input_data).predict
                    meta_features[idx] = pred[0]
                    mfs = dict(zip(meta_feature_names, meta_features[idx]))
                    if update_cached:
                        self._update_meta_features_cache(d, mfs)
        meta_features = pd.DataFrame.from_dict(meta_features, orient='index')
        return meta_features

    def _load_extractor(self):
        with IndustrialModels():
            pipeline = Pipeline().load(
                Path(get_project_root(), 'meta_automl', 'data_preparation', 'meta_features_extractors', 'time_series',
                     'extractor', '0_pipeline_saved', '0_pipeline_saved.json'))
            return pipeline
