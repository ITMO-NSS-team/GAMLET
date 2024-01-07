from gamlet.components.datasets_loaders.custom_datasets_loader import CustomDatasetsLoader
from gamlet.components.feature_preprocessors import FeaturesPreprocessor
from gamlet.components.meta_features_extractors import PymfeExtractor
from gamlet.data_preparation.file_system import get_configs_dir, get_data_dir
from gamlet.data_preparation.surrogate_dataset.dataset_generate import (KnowledgeBaseToDataset,
                                                                        dataset_from_id_without_data_loading)
from gamlet.surrogate.data_pipeline_surrogate import get_extractor_params


def main():
    datasets_loader = CustomDatasetsLoader(dataset_from_id_func=dataset_from_id_without_data_loading)
    extractor_params = get_extractor_params(get_configs_dir() / "use_features.json")

    converter = KnowledgeBaseToDataset(
        knowledge_base_directory=get_data_dir() / "knowledge_base_time_series_0/knowledge_base_time_series_0",
        dataset_directory=get_data_dir() / "timeseries",  # pymfe_meta_features_and_fedot_pipelines
        data_type="ts",
        datasets_loader=datasets_loader,
        meta_features_extractor=PymfeExtractor(**extractor_params),
        meta_features_preprocessor=FeaturesPreprocessor()
    )
    converter.convert_pipelines()
    # converter.convert_datasets()

if __name__ == "__main__":
    main()
