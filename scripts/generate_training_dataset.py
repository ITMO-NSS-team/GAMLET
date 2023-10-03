import argparse

import sys
import os
sys.path.append(os.getcwd())

from sklearn.preprocessing import StandardScaler, OneHotEncoder


from meta_automl.surrogate.datasets import KnowledgeBaseToDataset
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor


def main():
    preprocessors = {
        "MajorityClassSize": StandardScaler(),
        "MaxNominalAttDistinctValues": StandardScaler(),
        "MinorityClassSize": StandardScaler(),
        "NumberOfClasses": StandardScaler(),
        "NumberOfFeatures": StandardScaler(),
        "NumberOfInstances": StandardScaler(),
        "NumberOfInstancesWithMissingValues": StandardScaler(),
        "NumberOfMissingValues": StandardScaler(),
        "NumberOfNumericFeatures": StandardScaler(),
        "NumberOfSymbolicFeatures": StandardScaler(),
    }
    meta_features_preprocessor = FeaturesPreprocessor(preprocessors)


    converter = KnowledgeBaseToDataset(
        knowledge_base_directory = "data/knowledge_base_0",
        dataset_directory = "data/openml_meta_features_and_fedot_pipelines",
        meta_features_data_columns = [
            "MajorityClassSize",
            "MaxNominalAttDistinctValues",
            "MinorityClassSize",
            "NumberOfClasses",
            "NumberOfFeatures",
            "NumberOfInstances",
            "NumberOfInstancesWithMissingValues",
            "NumberOfMissingValues",
            "NumberOfNumericFeatures",
            "NumberOfSymbolicFeatures"
        ],
        train_test_split_name = "train_test_datasets_classification.csv",
        task_type="classification",
        fitness_metric = "f1",
        exclude_datasets = ["connect-4", "higgs"],
        meta_features_preprocessors=meta_features_preprocessor,
    )
    converter.convert()

if __name__ == '__main__':
    main()