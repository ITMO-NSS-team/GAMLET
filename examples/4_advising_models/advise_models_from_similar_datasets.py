from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from golem.core.optimisers.fitness import SingleObjFitness
from sklearn.model_selection import train_test_split

from meta_automl.data_preparation.dataset import DatasetCache
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import KNNSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseFEDOTPipelineAdvisor


def main():
    # Define datasets.
    dataset_names = ['monks-problems-1', 'apsfailure', 'australian', 'bank-marketing']
    # Extract meta-features and load on demand.
    extractor = PymfeExtractor(extractor_params={'groups': 'general'}, datasets_loader=OpenMLDatasetsLoader())
    meta_features = extractor.extract(dataset_names)
    # Preprocess meta-features, as KNN does not support NaNs.
    meta_features = meta_features.dropna(axis=1, how='any')
    # Split datasets to train (preprocessing) and test (actual meta-algorithm objects).
    x_train, x_test = train_test_split(meta_features, train_size=0.75, random_state=42)
    y_train = x_train.index
    assessor = KNNSimilarityAssessor({'n_neighbors': 2}, n_best=2)
    assessor.fit(x_train, y_train)
    # Define best models for datasets.
    best_pipelines = [
        PipelineBuilder().add_node('scaling').add_node('rf').to_pipeline(),
        PipelineBuilder().add_node('normalization').add_node('logit').to_pipeline(),
        PipelineBuilder().add_node('rf').add_node('logit').to_pipeline()
    ]
    best_models = [[Model(pipeline, SingleObjFitness(1), DatasetCache(dataset_name))]
                   for dataset_name, pipeline in zip(y_train, best_pipelines)]

    dataset_names_to_best_pipelines = dict(zip(y_train, best_models))
    advisor = DiverseFEDOTPipelineAdvisor(assessor, minimal_distance=2).fit(dataset_names_to_best_pipelines)
    return advisor.predict(x_test)


if __name__ == '__main__':
    result = main()
