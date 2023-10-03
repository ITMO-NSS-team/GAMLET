import argparse

import sys
import os
sys.path.append(os.getcwd())

from sklearn.preprocessing import StandardScaler, OneHotEncoder


from meta_automl.surrogate.datasets import KnowledgeBaseToDataset
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import OpenMLDatasetMetaFeaturesExtractor, PymfeExtractor
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader


def main():

    meta_features_extractor = PymfeExtractor(
        extractor_params={"groups": ["general", "statistical"]},                                
        datasets_loader=OpenMLDatasetsLoader()
    )
    # OpenMLDatasetMetaFeaturesExtractor(meta_features_data_columns=self.meta_features_data_columns)
            
    meta_features_preprocessor = FeaturesPreprocessor()
    
    converter = KnowledgeBaseToDataset(
        knowledge_base_directory = "data/knowledge_base_0",
        dataset_directory = "data/openml_meta_features_and_fedot_pipelines",
        meta_features_extractor = meta_features_extractor,
        train_test_split_name = "train_test_datasets_classification.csv",
        task_type="classification",
        fitness_metric = "f1",
        exclude_datasets = ["connect-4", "higgs"],
        meta_features_preprocessors=meta_features_preprocessor,
    )
    converter.convert()

if __name__ == '__main__':
    main()