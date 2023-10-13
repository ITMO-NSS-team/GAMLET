import os
from pathlib import Path

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.prediction_intervals.graph_distance import get_distance_between
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from meta_automl.data_preparation.dataset.time_series_dataset import TimeSeriesDataset
from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors.time_series.time_series_meta_features_extractor import \
    TimeSeriesFeaturesExtractor
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import KNeighborsBasedSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseFEDOTPipelineAdvisor


# TODO refactor example when there will a stable release

def dataset_to_pipelines(d_id):
    adapter = PipelineAdapter()
    dir_to_search = Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets', d_id)
    try:
        history = OptHistory().load(Path(dir_to_search, 'opt_history.json'))
    except Exception as e:
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
    dataset_names = os.listdir(Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets'))
    loader = TimeSeriesDatasetsLoader()
    datasets = loader.load(dataset_names)
    # Extract meta-features and load on demand.
    extractor = TimeSeriesFeaturesExtractor()
    meta_features = extractor.extract(datasets)
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
        pipeline1 = predict[i][0].predictor


        # pipeline.show()
        # dataset_names_to_best_pipelines_test[idx].predictor.show()
        # pipeline = predict[np.random.choice(np.arange(len(predict)))][0].predictor
        #dists.append(get_distance_between(pipeline, dataset_names_to_best_pipelines_test[idx].predictor))

        pipeline1.unfit()
        pipeline1.fit(train_data)
        pred = np.ravel(pipeline1.predict(test_data).predict)

        if len(predict[i]) > 1:
            pipeline2 = predict[i][1].predictor
            pipeline2.unfit()
            pipeline2.fit(train_data)
            pred2 = np.ravel(pipeline2.predict(test_data).predict)

            pred = (pred + pred2) / 2
        df = pd.DataFrame({'value': y, 'predict': pred})
        df.to_csv(f'res/{idx}_forecast_vs_actual.csv')

    print(np.array(dists).mean())


if __name__ == '__main__':
    main()
