import os

import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from sklearn.metrics import mean_absolute_error

from gamlet.utils import project_root
from rl_core.dataloader import TimeSeriesDataLoader
from rl_core.utils import define_data_for_experiment

PIPELINES_WITHOUT_LAGGED = [
    'ar',
    'polyfit',
    'glm',
    'ets',
    'locf'
]

PIPELINES_WITH_LAGGED = [
    'adareg',
    'dtreg',
    'lasso',
    'lgbmreg',
    'rfr',
    'ridge',
    'sgdr',
    'svr',
    'ts_naive_average',
]

if __name__ == '__main__':
    """
        This script is designed to create a DataFrame with metric values for different M4 datasets.
        It step-by-step builds a simple pipeline with a single node and with «lagged ->» pipelines.
        Then it creates pipelines based on the composing experiments called «base» and using topological models.
        The resulting DataFrame allows us to visualize the space of pipeline solutions.
    """
    columns = ['Dataset'] + PIPELINES_WITHOUT_LAGGED + \
              [f'lagged -> {p}' for p in PIPELINES_WITH_LAGGED] + \
              ['Base Pipeline', 'Topo Pipeline']

    result = pd.DataFrame([], columns=columns)

    data_folder_path = os.path.join(str(project_root()), 'MetaFEDOT\\data\\knowledge_base_time_series_0\\datasets\\')
    dataset_names = [name for name in os.listdir(data_folder_path)]

    dataloader, _, train_list, _ = define_data_for_experiment()

    train_datasets = {}
    for dataset in train_list:
        train_datasets[dataset] = os.path.join(data_folder_path, f'{dataset}/data.csv')

    dataloader = TimeSeriesDataLoader(train_datasets)

    single_pipeline_len = len(PIPELINES_WITH_LAGGED) + len(PIPELINES_WITHOUT_LAGGED)

    for dataset in train_list:
        print(f'-- {dataset} started -- ')
        train_data, test_data, meta_data = dataloader.get_data(dataset_name=dataset)

        df_col = {
            'Dataset': dataset
        }

        y_true = test_data.target

        for pipeline in PIPELINES_WITHOUT_LAGGED:
            pipeline_ = PipelineBuilder().add_node(pipeline).build()

            try:
                pipeline_.fit(train_data)

                y_pred = pipeline_.predict(test_data).predict

                metric = mean_absolute_error(y_true, y_pred)

                df_col[pipeline] = metric

            except:
                metric = np.nan

            print(f'-- {pipeline} : {metric} -- ')

        for pipeline in PIPELINES_WITH_LAGGED:
            pipeline_ = PipelineBuilder().add_node('lagged').add_node(pipeline).build()

            try:
                pipeline_.fit(train_data)

                y_pred = pipeline_.predict(test_data).predict

                metric = mean_absolute_error(y_true, y_pred)

                df_col[f'lagged -> {pipeline}'] = metric

            except:
                metric = np.nan

            print(f'-- lagged -> {pipeline} : {metric} -- ')

        path_to_base = os.path.join(
            str(project_root()),
            f'MetaFEDOT\\data\\knowledge_base_time_series_0\\datasets\\{dataset}\\'
            f'model\\0_pipeline_saved\\0_pipeline_saved.json'
        )

        try:
            base_pipeline_ = Pipeline().load(source=path_to_base)
            y_pred = base_pipeline_.predict(test_data).predict

            metric = mean_absolute_error(y_true, y_pred)

            df_col['Base Pipeline'] = metric

        except:
            metric = np.nan

        print(f'-- base pipeline : {metric} -- ')

        try:
            path_to_topo = os.path.join(
                str(project_root()),
                f'MetaFEDOT\\data\\topo_ws_selection_evo\\{dataset.split("_")[1]}\\'
                f'model\\0_pipeline_saved\\0_pipeline_saved.json'
            )

            topo_pipeline_ = Pipeline().load(source=path_to_topo)
            y_pred = topo_pipeline_.predict(test_data).predict

            metric = mean_absolute_error(y_true, y_pred)

        except:
            metric = np.nan

        df_col['Topo Pipeline'] = metric

        print(f'-- topo pipeline : {metric} -- ')

        temp = pd.DataFrame([df_col])
        result = pd.concat([result, temp], ignore_index=True)

    result.to_csv('result.csv')
