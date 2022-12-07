from __future__ import annotations

from typing import List, Union

import openml
import pandas as pd

from meta_automl.data_preparation.dataset import DatasetCache
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor


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
