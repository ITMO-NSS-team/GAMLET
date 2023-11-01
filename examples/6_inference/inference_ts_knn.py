import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from golem.core.dag.linked_graph import get_distance_between
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

from meta_automl.data_preparation.dataset.time_series_dataset import TimeSeriesDataset
from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.evaluated_model import EvaluatedModel
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors.time_series.time_series_meta_features_extractor import \
    TimeSeriesFeaturesExtractor
from meta_automl.meta_algorithm.dataset_similarity_assessors import KNeighborsSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseModelAdvisor


# TODO refactor example when there will a stable release

def dataset_to_models(d_id):
    adapter = PipelineAdapter()
    d_id = str(d_id)
    dir_to_search = Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets', d_id)
    try:
        history = OptHistory().load(Path(dir_to_search, 'opt_history.json'))
    except Exception:
        return None

    best_fitness = 1000000
    best_model = None

    for gen in history.generations:
        for ind in gen:
            if ind.fitness.value < best_fitness:
                pipeline = adapter.restore(ind.graph)
                best_model = EvaluatedModel(pipeline, ind.fitness.value, history.objective.metric_names[0],
                                            TimeSeriesDataset(d_id))
                best_fitness = ind.fitness.value
    return [best_model]


def main():
    dataset_names = os.listdir(Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets'))
    loader = TimeSeriesDatasetsLoader()
    datasets = loader.load(dataset_names)
    # Extract meta-features and load on demand.
    extractor = TimeSeriesFeaturesExtractor()
    meta_features = extractor.extract(datasets)
    meta_features = meta_features.dropna(axis=1, how='any')
    # Split datasets to train (preprocessing) and test (actual meta-algorithm objects).
    mf_train, mf_test = train_test_split(meta_features, train_size=0.75, random_state=42)
    did_train = mf_train.index
    did_test = mf_test.index
    # Dimension reduction
    q_t = QuantileTransformer(output_distribution='normal')
    mf_train_v = q_t.fit_transform(mf_train)
    mf_test_v = q_t.transform(mf_test)
    mf_train = pd.DataFrame(data=mf_train_v, index=mf_train.index)
    mf_test = pd.DataFrame(data=mf_test_v, index=mf_test.index)

    # Define best models for datasets.
    dataset_ids_to_best_models = {}
    for d_id in tqdm(did_train):
        models = dataset_to_models(d_id)
        if models is not None:
            dataset_ids_to_best_models[d_id] = models
    did_train, best_models_train = zip(*dataset_ids_to_best_models.items())
    mf_train = mf_train[mf_train.index.isin(did_train)]
    assessor = KNeighborsSimilarityAssessor(n_neighbors=2)
    assessor.fit(mf_train, did_train)
    advisor = DiverseModelAdvisor(minimal_distance=2).fit(did_train, best_models_train)
    did_pred = assessor.predict(mf_test)
    best_models_pred = advisor.predict(did_pred)
    forecast_length = {'D': 14, 'W': 13, 'M': 18, 'Q': 8, 'Y': 6}
    dataset_names_to_best_models_test = {}
    for d_id in tqdm(did_test):
        models = dataset_to_models(d_id)
        if models is not None:
            dataset_names_to_best_models_test[d_id] = models

    dists = []
    dists_random = []
    for dataset_id, models in tqdm(zip(did_test, best_models_pred), 'Test datasets', len(did_test)):
        if dataset_id not in dataset_names_to_best_models_test:
            continue
        # Define datasets.
        loader = TimeSeriesDatasetsLoader(forecast_length=forecast_length[dataset_id[3]])
        dataset = loader.load_single(dataset_id)
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
        pipeline = models[0].predictor

        pipeline_random = random.choice(list(dataset_names_to_best_models_test.values())[0]).predictor

        dists.append(get_distance_between(pipeline, dataset_names_to_best_models_test[dataset_id][0].predictor))
        dists_random.append(
            get_distance_between(pipeline_random, dataset_names_to_best_models_test[dataset_id][0].predictor))
        # #
        pipeline.unfit()
        pipeline.fit(train_data)
        pred = np.ravel(pipeline.predict(test_data).predict)

        df = pd.DataFrame({'value': y, 'predict': pred})
        df.to_csv(f'{dataset_id}_forecast_vs_actual.csv')

    print(f'Dist mean: {np.array(dists).mean()}')
    print(f'Dist random mean: {np.array(dists_random).mean()}')


if __name__ == '__main__':
    main()
