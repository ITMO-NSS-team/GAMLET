import logging
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from tqdm.auto import tqdm

from meta_automl.data_preparation.dataset import DatasetBase, DatasetIDType
from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor


class TimeSeriesFeaturesExtractor(MetaFeaturesExtractor):
    default_params: Optional[Dict[str, Any]] = None

    def __init__(self):
        self._extractor = self._generate_extractor()
        self._datasets_loader = TimeSeriesDatasetsLoader()

    def extract(self,
                datasets_or_ids: List[Union[DatasetBase, DatasetIDType]],
                fill_input_nans: bool = False, use_cached: bool = True, update_cached: bool = True,
                fit_kwargs: Dict[str, Any] = None, extract_kwargs: Dict[str, Any] = None) -> pd.DataFrame:

        meta_features = {}
        for dataset in datasets_or_ids:
            if not isinstance(dataset, DatasetBase):
                dataset = self._datasets_loader.load_single(dataset)

        logging.critical(f'Extracting meta features of the time series datasets...')

        with IndustrialModels():
            for d in tqdm(datasets_or_ids, desc='Feature_generation'):
                idx = d.id_
                features = d.get_data().x
                input_data = InputData(idx=np.array([0]), features=np.array(features), target=None,
                                       task=Task(TaskTypesEnum.classification),
                                       data_type=DataTypesEnum.table)
                meta_features[idx] = self._extractor.root_node.predict(input_data).predict[0]

        meta_features = pd.DataFrame.from_dict(meta_features, orient='index')
        return meta_features

    def _generate_extractor(self):
        with IndustrialModels():
            pipeline = PipelineBuilder() \
                .add_node('wavelet_basis') \
                .add_branch('quantile_extractor', 'topological_extractor') \
                .add_node('quantile_extractor', branch_idx=4) \
                .add_node('topological_extractor', branch_idx=5) \
                .join_branches('cat_features') \
                .build()
            # dummy fit (we need pipeline state be fitt)
            dummy_data = InputData(idx=np.array(np.arange(1)), features=np.array(
                [np.random.random(30) for _ in range(2)]),
                                   target=None,
                                   task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
            pipeline._fit(dummy_data)
            return pipeline
