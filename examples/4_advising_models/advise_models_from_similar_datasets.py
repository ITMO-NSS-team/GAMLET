from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from golem.core.optimisers.fitness import SingleObjFitness
from sklearn.model_selection import train_test_split

from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.evaluated_model import EvaluatedModel
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.meta_algorithm.dataset_similarity_assessors import KNeighborsSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseModelAdvisor


def main():
    # Define datasets.
    dataset_names = ['monks-problems-1', 'apsfailure', 'australian', 'bank-marketing']
    datasets = OpenMLDatasetsLoader().load(dataset_names, allow_names=True)
    # Extract meta-features and load on demand.
    extractor = PymfeExtractor(extractor_params={'groups': 'general'})
    meta_features = extractor.extract(datasets)
    # Preprocess meta-features, as KNN does not support NaNs.
    meta_features = meta_features.dropna(axis=1, how='any')
    dataset_ids = meta_features.index
    # Split datasets to train (preprocessing) and test (actual meta-algorithm objects).
    mf_train, mf_test, did_train, did_test = train_test_split(meta_features, dataset_ids, train_size=0.75,
                                                              random_state=42)

    # Define best models for datasets.
    best_pipelines = [
        PipelineBuilder().add_node('scaling').add_node('rf').build(),
        PipelineBuilder().add_node('normalization').add_node('logit').build(),
        PipelineBuilder().add_node('rf').add_node('logit').build()
    ]
    best_models_train = [[EvaluatedModel(pipeline, SingleObjFitness(1), 'some_metric_name', OpenMLDataset(dataset_id))]
                         for dataset_id, pipeline in zip(did_train, best_pipelines)]

    # Train the component that calculates similarity between datasets
    assessor = KNeighborsSimilarityAssessor(n_neighbors=2).fit(mf_train, did_train)
    # Train the component remembers best models for datasets
    advisor = DiverseModelAdvisor(minimal_distance=2).fit(dataset_ids=did_train, models=best_models_train)
    # Predict similar datasets from train
    did_pred = assessor.predict(mf_test)
    # Predict models for similar datasets
    return advisor.predict(dataset_ids=did_pred)


if __name__ == '__main__':
    result = main()
    print(result)
