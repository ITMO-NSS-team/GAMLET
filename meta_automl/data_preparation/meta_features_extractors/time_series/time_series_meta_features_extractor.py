from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from tqdm import tqdm

from meta_automl.data_preparation.dataset import TimeSeriesData, TimeSeriesDataset
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor

TS_FEATURES_COUNT = 184


class TimeSeriesFeaturesExtractor(MetaFeaturesExtractor):
    default_params: Optional[Dict[str, Any]] = None

    def __init__(self):
        self._extractor = self._load_extractor()
        self.meta_feature_names = np.arange(TS_FEATURES_COUNT)

    def extract(self, datasets_or_ids: Sequence[Union[TimeSeriesDataset, TimeSeriesData]]) -> pd.DataFrame:

        rows = {}

        for dataset_data in tqdm(datasets_or_ids, desc='Extracting meta features of the datasets'):
            if isinstance(dataset_data, TimeSeriesDataset):
                dataset_data = dataset_data.get_data()

            features = dataset_data.x

            input_data = InputData(idx=np.array([0]), features=np.array(features).reshape(1, -1), target=None,
                                   task=Task(TaskTypesEnum.classification),
                                   data_type=DataTypesEnum.table)
            pred = self._extractor.root_node.predict(input_data).predict
            meta_features_extracted = pred[0]
            meta_features_extracted = dict(zip(self.meta_feature_names, meta_features_extracted))
            rows[dataset_data.id] = meta_features_extracted

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
