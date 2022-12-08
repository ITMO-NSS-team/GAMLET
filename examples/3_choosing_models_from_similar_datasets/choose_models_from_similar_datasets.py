from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from sklearn.model_selection import train_test_split

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.meta_algorithm.datasets_similarity_assessors import KNNSimilarityAssessor


def main():
    # Define datasets.
    dataset_names = ['amazon_employee_access', 'apsfailure', 'australian', 'bank-marketing']
    # Extract meta-features and load on demand.
    extractor = PymfeExtractor().fit(extractor_params={'groups': 'general'}, datasets_loader=OpenMLDatasetsLoader())
    meta_features = extractor.extract(dataset_names)
    # Preprocess meta-features, as KNN does not support NaNs.
    meta_features = meta_features.dropna(axis=1, how='any')
    # Split datasets to train (preprocessing) and test (actual meta-algorithm objects).
    x_train, x_test = train_test_split(meta_features, train_size=0.75, random_state=42)
    # Could use any of the classes ``ModelSelector`` but use synthetic pipelines for the example.
    y_train = [PipelineBuilder().add_node('scaling').add_node('rf').to_pipeline(),
               PipelineBuilder().add_node('normalization').add_node('knn').to_pipeline(),
               PipelineBuilder().add_node('rf').add_node('logit').to_pipeline()]
    assessor = KNNSimilarityAssessor({'n_neighbors': 1}, n_best=1)
    assessor.fit(x_train, y_train)
    # Get models for the best fitting datasets from train.
    return assessor.get_similar(x_test)


if __name__ == '__main__':
    result = main()
