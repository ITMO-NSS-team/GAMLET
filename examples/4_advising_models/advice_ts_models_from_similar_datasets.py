import os
from pathlib import Path

from fedot.core.pipelines.adapters import PipelineAdapter
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from meta_automl.data_preparation.dataset.time_series_dataset import TimeSeriesDataset
from meta_automl.data_preparation.datasets_loaders.timeseries_dataset_loader import TimeSeriesDatasetsLoader
from meta_automl.data_preparation.evaluated_model import EvaluatedModel
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors.time_series.time_series_meta_features_extractor import \
    TimeSeriesFeaturesExtractor
from meta_automl.meta_algorithm.dataset_similarity_assessors import KNeighborsSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseModelAdvisor


def dataset_to_models(d_id):
    adapter = PipelineAdapter()
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
    # Define datasets.
    dataset_names = os.listdir(Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets'))
    loader = TimeSeriesDatasetsLoader()
    datasets = loader.load(dataset_names)
    # Preprocess meta-features, as KNN does not support NaNs.

    extractor = TimeSeriesFeaturesExtractor()
    meta_features = extractor.extract(datasets)
    meta_features = meta_features.dropna(axis=1, how='any')
    dataset_ids = meta_features.index

    # Split datasets to train (preprocessing) and test (actual meta-algorithm objects).
    mf_train, mf_test, did_train, did_test = train_test_split(meta_features, dataset_ids, train_size=0.75,
                                                              random_state=42)

    # Define best models for datasets.
    dataset_ids_to_best_models = {}
    for d_id in tqdm(did_train, 'Loading models for train datasets'):
        best_models_train = dataset_to_models(d_id)
        if best_models_train is not None:
            dataset_ids_to_best_models[d_id] = best_models_train
    mf_train = mf_train[mf_train.index.isin(dataset_ids_to_best_models.keys())]
    did_train = did_train[did_train.isin(dataset_ids_to_best_models.keys())]
    dataset_ids_train, best_models_train = zip(*dataset_ids_to_best_models.items())

    # Train the component that calculates similarity between datasets
    assessor = KNeighborsSimilarityAssessor(n_neighbors=2).fit(mf_train, did_train)
    # Train the component remembers best models for datasets
    advisor = DiverseModelAdvisor(minimal_distance=2).fit(dataset_ids_train, best_models_train)
    # Predict similar datasets from train
    did_pred = assessor.predict(mf_test)
    # Predict models for similar datasets
    return advisor.predict(did_pred)


if __name__ == '__main__':
    result = main()
    print(result)
