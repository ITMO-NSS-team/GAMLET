import os
from pathlib import Path

from fedot.core.pipelines.adapters import PipelineAdapter
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
    # Define datasets.
    dataset_names = os.listdir(Path(get_project_root(), 'data', 'knowledge_base_time_series_0', 'datasets'))
    loader = TimeSeriesDatasetsLoader()
    datasets = loader.load(dataset_names)
    # Preprocess meta-features, as KNN does not support NaNs.

    extractor = TimeSeriesFeaturesExtractor()
    meta_features = extractor.extract(datasets)
    meta_features = meta_features.dropna(axis=1, how='any')

    # Split datasets to train (preprocessing) and test (actual meta-algorithm objects).
    x_train, x_test = train_test_split(meta_features, train_size=0.75, random_state=42)
    y_train = x_train.index

    # Define best models for datasets.
    dataset_names_to_best_pipelines = {}
    for d_id in tqdm(y_train):
        if dataset_to_pipelines(d_id) is not None:
            dataset_names_to_best_pipelines[d_id] = dataset_to_pipelines(d_id)
    x_train = x_train[x_train.index.isin(dataset_names_to_best_pipelines.keys())]
    y_train = y_train[y_train.isin(dataset_names_to_best_pipelines.keys())]
    assessor = KNeighborsBasedSimilarityAssessor(n_neighbors=2)
    assessor.fit(x_train, y_train)
    advisor = DiverseFEDOTPipelineAdvisor(assessor, minimal_distance=2).fit(dataset_names_to_best_pipelines)
    return advisor.predict(x_test)


if __name__ == '__main__':
    result = main()
    print(result)
