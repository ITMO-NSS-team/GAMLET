"""Simple example on advising pipelines for a dataset.

Run this model from the repo root.
"""

import pickle
from pathlib import Path

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.data_preparation.file_system.file_system import get_configs_dir, get_project_root
from meta_automl.meta_algorithm.model_advisors import SurrogateGNNModelAdvisor
from meta_automl.surrogate.data_pipeline_surrogate import get_extractor_params
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel

SURROGATE_MODEL_CHECKPOINT_FILE = "./experiments/base/checkpoints/best.ckpt"
SURROGATE_MODEL_HYPERPARAMETERS_FILE = "./experiments/base/hparams.yaml"
META_FEATURES_EXTRACTOR_CONFIG_FILE = "./configs/use_features.json"
META_FEATURES_PREPROCESSOR_FILE = "./data/pymfe_meta_features_and_fedot_pipelines/all/meta_features_preprocessors.pickle"
PIPELINES_FILE = "./data/pymfe_meta_features_and_fedot_pipelines/all/pipelines_fedot.pickle"


def main():
    # Build model to estimate pipelines.
    surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
        checkpoint_path=SURROGATE_MODEL_CHECKPOINT_FILE,
        hparams_file=SURROGATE_MODEL_HYPERPARAMETERS_FILE,
    )
    # Build dataset meta features extractor.
    extractor_params = get_extractor_params(META_FEATURES_EXTRACTOR_CONFIG_FILE)
    dataset_meta_features_extractor = PymfeExtractor(extractor_params)
    # Build dataset meta features preprocessor.
    dataset_meta_features_preprocessor = FeaturesPreprocessor(
        load_path=META_FEATURES_PREPROCESSOR_FILE, extractor_params=extractor_params
    )
    # Build pipeline features extractor.
    pipeline_extractor = FEDOTPipelineFeaturesExtractor(
        include_operations_hyperparameters=False,
        operation_encoding="ordinal",
    )
    # Build adviser.
    advisor = SurrogateGNNModelAdvisor(
        surrogate_model,
        dataset_meta_features_extractor,
        dataset_meta_features_preprocessor,
        pipeline_extractor,
    )
    # Load datasets.
    datasets_names = [
        "apsfailure",
    ]
    datasets = [OpenMLDatasetsLoader().load([dset_name], allow_names=True) for dset_name in datasets_names]
    # Load pipelines.
    with open(PIPELINES_FILE, "rb") as input_file:
        pipelines = pickle.load(input_file)
    # Make prediction.
    return advisor.predict(pipelines, datasets, k=5)


if __name__ == "__main__":
    result = main()
    print(result)
