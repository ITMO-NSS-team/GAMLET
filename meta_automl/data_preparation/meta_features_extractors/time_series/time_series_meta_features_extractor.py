from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from tqdm import tqdm

from meta_automl.data_preparation.dataset import DatasetBase, DatasetIDType
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader, TimeSeriesDatasetsLoader
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor

TS_FEATURES_COUNT = 184


class TimeSeriesFeaturesExtractor(MetaFeaturesExtractor):
    default_params: Optional[Dict[str, Any]] = None

    def __init__(self, custom_path=None):
        self._extractor = self._load_extractor()
        self._datasets_loader = TimeSeriesDatasetsLoader(custom_path=custom_path)
        self.meta_feature_names = np.arange(TS_FEATURES_COUNT)

    @property
    def datasets_loader(self) -> DatasetsLoader:
        if not self._datasets_loader:
            raise ValueError("Datasets loader not provided!")
        return self._datasets_loader

    def extract(self,
                datasets_or_ids: List[Union[DatasetBase, DatasetIDType]], use_cached: bool = True,
                update_cached: bool = True) -> pd.DataFrame:

        rows = {}

        for dataset in tqdm(datasets_or_ids, desc='Extracting meta features of the datasets'):
            if isinstance(dataset, DatasetBase):
                dataset_id = dataset.id_
                dataset_class = dataset.__class__
            else:
                dataset_id = dataset
                dataset_class = self.datasets_loader.dataset_class
            meta_features_cached = self._get_meta_features_cache(dataset_id, dataset_class, self.meta_feature_names)

            if use_cached and meta_features_cached:
                rows[dataset_id] = meta_features_cached
            else:
                if not isinstance(dataset, DatasetBase):
                    dataset = self._datasets_loader.load_single(dataset)
                features = dataset.get_data().x

                input_data = InputData(idx=np.array([0]), features=np.array(features).reshape(1, -1), target=None,
                                       task=Task(TaskTypesEnum.classification),
                                       data_type=DataTypesEnum.table)
                with IndustrialModels():
                    pred = self._extractor.root_node.predict(input_data).predict
                meta_features_extracted = pred[0]
                meta_features_extracted = dict(zip(self.meta_feature_names, meta_features_extracted))
                rows[dataset_id] = meta_features_extracted
                if update_cached:
                    self._update_meta_features_cache(dataset_id, dataset_class, meta_features_extracted)

        meta_features = pd.DataFrame.from_dict(rows, orient='index')

        return meta_features

    @staticmethod
    def _load_extractor():
        path = get_project_root().joinpath('meta_automl', 'data_preparation', 'meta_features_extractors',
                                           'time_series', 'extractor', '0_pipeline_saved', '0_pipeline_saved.json')
        if not path.exists():
            raise ValueError('Pretrained data is not loaded.')
        path = str(path)
        with IndustrialModels():
            pipeline = Pipeline().load(path)
        return pipeline
