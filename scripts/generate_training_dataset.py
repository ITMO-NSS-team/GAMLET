from meta_automl.data_preparation.datasets_loaders.custom_datasets_loader import CustomDatasetsLoader
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.surrogate_dataset.dataset_generate import KnowledgeBaseToDataset, \
    dataset_from_id_without_data_loading
from meta_automl.surrogate.data_pipeline_surrogate import get_extractor_params


def main():
    datasets_loader = CustomDatasetsLoader(dataset_from_id_func=dataset_from_id_without_data_loading)
    extractor_params = get_extractor_params('configs/use_features.json')

    meta_features_extractor = PymfeExtractor(
        extractor_params=extractor_params,
        datasets_loader=datasets_loader,
    )

    meta_features_preprocessor = FeaturesPreprocessor(extractor_params=extractor_params)

    converter = KnowledgeBaseToDataset(
        knowledge_base_directory="data/knowledge_base_1",
        dataset_directory="data/pymfe_meta_features_and_fedot_pipelines",
        meta_features_extractor=meta_features_extractor,
        train_test_split_name="train_test_datasets_classification.csv",
        task_type="classification",
        fitness_metric="fitness",
        meta_features_preprocessors=meta_features_preprocessor,
        models_loader_kwargs={"datasets_loader": datasets_loader}
    )
    converter.convert_pipelines()
    converter.convert_datasets()


if __name__ == '__main__':
    main()
