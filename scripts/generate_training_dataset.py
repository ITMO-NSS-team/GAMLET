import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
from typing import Dict

sys.path.append(os.getcwd())
from meta_automl.data_preparation.dataset import (CustomDataset,
                                                  DataNotFoundError,
                                                  DatasetData, DatasetIDType)
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.datasets_loaders.custom_datasets_loader import CustomDatasetsLoader
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import (OpenMLDatasetMetaFeaturesExtractor,
                                                                   PymfeExtractor)
from meta_automl.surrogate.datasets import KnowledgeBaseToDataset


def dataset_from_id_without_data_loading(dataset_id: DatasetIDType) -> CustomDataset:
    """ Creates the CustomDataset object without loading the data. Use if your don't need the models
    to load the datasets data into memory, or if you have loaded the cache manually. """
    return CustomDataset(dataset_id)

def dataset_from_id_with_data_loading(dataset_id: DatasetIDType) -> CustomDataset:
    """ Load dataset from '//10.9.14.114/calc/Nikitin/datasets/' into the project cache directory.
    As a result, every model of the knowledge base will have its data available by
    `model.dataset.get_data()`.
    """
    dataset = CustomDataset(dataset_id)
    try:
        dataset.get_data()
    except DataNotFoundError:
        data_root = '//10.9.14.114/calc/Nikitin/datasets/'
        dataset_name, fold_num = dataset_id[:-2], dataset_id[-1]
        data_path = f'{dataset_name}_fold{fold_num}.npy'
        data_x = []
        for path_prefix in ('train_', 'test_'):
            data_x.append(np.load(data_root + path_prefix + data_path))
        data_y = []
        for path_prefix in ('trainy_', 'testy_'):
            data_y.append(np.load(data_root + path_prefix + data_path))
        data_x = np.concatenate(data_x)
        data_y = np.concatenate(data_y)
        data_x = pd.DataFrame(data_x)
        data_y = pd.DataFrame(data_y)
        dataset_data = DatasetData(data_x, data_y)
        dataset.dump_data(dataset_data)
    return dataset

def get_extractor_params(filename: str) -> Dict[str, str]:
    with open(filename) as f:
        extractor_params = json.load(f)
    return extractor_params

def main():
    # datasets_loader_builder = lambda: CustomDatasetsLoader(dataset_from_id_func=dataset_from_id_with_data_loading)
    datasets_loader_builder = lambda: CustomDatasetsLoader(dataset_from_id_func=dataset_from_id_without_data_loading)
    extractor_params = get_extractor_params('scripts/use_features.json')

    meta_features_extractor = PymfeExtractor(
        extractor_params = extractor_params,
        datasets_loader = datasets_loader_builder(),
    )
    # OpenMLDatasetMetaFeaturesExtractor(meta_features_data_columns=self.meta_features_data_columns)

    meta_features_preprocessor = FeaturesPreprocessor()

    converter = KnowledgeBaseToDataset(
        knowledge_base_directory = "data/knowledge_base_1_v2",
        dataset_directory = "data/pymfe_meta_features_and_fedot_pipelines",
        meta_features_extractor = meta_features_extractor,
        train_test_split_name = "train_test_datasets_classification.csv",
        task_type="classification",
        fitness_metric = "fitness",
        meta_features_preprocessors=meta_features_preprocessor,
        models_loader_kwargs={"datasets_loader": datasets_loader_builder()}
    )
    converter.convert()

if __name__ == '__main__':
    main()