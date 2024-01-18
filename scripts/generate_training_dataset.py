import argparse

from gamlet.components.datasets_loaders import CustomDatasetsLoader, TimeSeriesDatasetsLoader
from gamlet.components.feature_preprocessors import FeaturesPreprocessor
from gamlet.components.meta_features_extractors import PymfeExtractor
from gamlet.components.models_loaders.knowledge_base_models_loader import CompatKBModelsLoader, KBTSModelsLoader
from gamlet.data_preparation.file_system import get_configs_dir, get_data_dir
from gamlet.data_preparation.surrogate_dataset.dataset_generate import (KnowledgeBaseToDataset,
                                                                        dataset_from_id_without_data_loading)
from gamlet.surrogate.data_pipeline_surrogate import get_extractor_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True)
    args = parser.parse_args()

    extractor_params = get_extractor_params(get_configs_dir() / "use_features.json")

    if args.data_type == "table":
        kb_dir = "knowledge_base_1"
        out_dir = "pymfe_meta_features_and_fedot_pipelines"
        datasets_loader = CustomDatasetsLoader(dataset_from_id_func=dataset_from_id_without_data_loading)
        models_loader = CompatKBModelsLoader(get_data_dir() / kb_dir, datasets_loader)
    elif args.data_type == "ts":
        kb_dir = "knowledge_base_time_series_0/knowledge_base_time_series_0"
        out_dir = "timeseries"
        datasets_loader = TimeSeriesDatasetsLoader()
        models_loader = KBTSModelsLoader(get_data_dir() / kb_dir, datasets_loader)
    else:
        raise ValueError("data_type should be 'table' or 'ts'")

    converter = KnowledgeBaseToDataset(
        kb_dir=get_data_dir() / kb_dir,
        dataset_directory=get_data_dir() / out_dir,
        models_loader=models_loader,
        datasets_loader=datasets_loader,
        meta_features_extractor=PymfeExtractor(**extractor_params),
        meta_features_preprocessor=FeaturesPreprocessor()
    )
    converter.convert_pipelines()
    converter.convert_datasets()


if __name__ == "__main__":
    main()
