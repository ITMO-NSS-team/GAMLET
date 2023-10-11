import os
from pathlib import Path

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.prediction_intervals.graph_distance import get_distance_between
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.tuning.simultaneous import SimultaneousTuner
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from meta_automl.data_preparation.dataset.time_series_dataset import TimeSeriesDataset
from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import KNeighborsBasedSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseFEDOTPipelineAdvisor


# TODO refactor example when there will a stable release

def dataset_to_pipelines(d_id):
    adapter = PipelineAdapter()
    dir_to_search = Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets', d_id)
    try:
        history = OptHistory().load(Path(dir_to_search, 'opt_history.json'))
    except:
        return None

    best_fitness = 1000000
    best_model = None

    for gen in history.generations:
        for ind in gen:
            if ind.fitness.value < best_fitness:
                pipeline = adapter.restore(ind.graph)
                best_model = Model(pipeline, ind.fitness.value, history.objective.metric_names[0],
                                   TimeSeriesDataset(d_id))
                best_fitness = ind.fitness.value
    return best_model


def main():

    meta_features = pd.read_csv('../../data/knowledge_base_time_series_0/meta_features_ts.csv', index_col=0)
    meta_features = meta_features.dropna(axis=1, how='any')
    idx = meta_features.index.values
    # Split datasets to train (preprocessing) and test (actual meta-algorithm objects).
    x_train, x_test = train_test_split(meta_features, train_size=0.75, random_state=42)
    y_train = x_train.index
    y_test = x_test.index

    # Define best models for datasets.
    dataset_names_to_best_pipelines = {}
    for d_id in tqdm(y_train):
        model = dataset_to_pipelines(d_id)
        if model is not None:
            dataset_names_to_best_pipelines[d_id] = model
    x_train = x_train[x_train.index.isin(dataset_names_to_best_pipelines.keys())]
    y_train = y_train[y_train.isin(dataset_names_to_best_pipelines.keys())]
    assessor = KNeighborsBasedSimilarityAssessor(n_neighbors=2)
    assessor.fit(x_train, y_train)
    advisor = DiverseFEDOTPipelineAdvisor(assessor, minimal_distance=2).fit(dataset_names_to_best_pipelines)
    predict = advisor.predict(x_test)

    forecast_length = {'D': 14, 'W': 13, 'M': 18, 'Q': 8, 'Y': 6}
    dataset_names_to_best_pipelines_test = {}
    for d_id in tqdm(y_test):
        model = dataset_to_pipelines(d_id)
        if model is not None:
            dataset_names_to_best_pipelines_test[d_id] = model

    dists = []
    for i in tqdm(range(len(x_test))):

        idx = x_test.index[i]
        # Define datasets.
        loader = TimeSeriesDatasetsLoader(forecast_length=forecast_length[idx[3]])
        dataset = loader.load([idx])[0]
        # Preprocess meta-features, as KNN does not support NaNs.
        data = dataset.get_data()
        X = data.x
        y = data.y
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len(y)))
        train_data = InputData(idx=np.arange(len(X)), features=X, target=X, task=task,
                               data_type=DataTypesEnum.ts)

        test_data = InputData(idx=np.arange(len(X)), features=X, target=y, task=task,
                              data_type=DataTypesEnum.ts)
        print(idx)
        pipeline = predict[i][0].predictor
        pipeline.show()
        dataset_names_to_best_pipelines_test[idx].predictor.show()
        #  pipeline = predict[np.random.choice(np.arange(len(predict)))][0].predictor
        dists.append(get_distance_between(pipeline, dataset_names_to_best_pipelines_test[idx].predictor))

        # pipeline.unfit()
        # tuner = TunerBuilder(task) \
        #     .with_tuner(SimultaneousTuner) \
        #     .with_metric(RegressionMetricsEnum.MAE) \
        #     .with_iterations(5) \
        #     .build(train_data)
        # pipeline = tuner.tune(pipeline)
        # pred = np.ravel(pipeline.predict(test_data).predict)
        # df = pd.DataFrame({'value': y, 'predict': pred})
        # df.to_csv(f'res/{idx}_forecast_vs_actual.csv')

    print(np.array(dists).mean())

if __name__ == '__main__':
    main()
