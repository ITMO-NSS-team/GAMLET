"""Simple example on advising pipelines for a dataset.

Run this model from the repo root.
"""

import pickle

from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.file_system.file_system import get_checkpoints_dir, get_configs_dir, get_data_dir
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.meta_algorithm.model_advisors import SurrogateGNNModelAdvisor
from meta_automl.surrogate.data_pipeline_surrogate import get_extractor_params
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel

SURROGATE_MODEL_CHECKPOINT_FILE = get_checkpoints_dir() / "tabular" / "checkpoints" / "best.ckpt"
SURROGATE_MODEL_HYPERPARAMETERS_FILE = get_checkpoints_dir() / "tabular" / "hparams.yaml"
META_FEATURES_EXTRACTOR_CONFIG_FILE = get_configs_dir() / "use_features.json"
META_FEATURES_PREPROCESSOR_FILE = (get_data_dir() /
                                   "pymfe_meta_features_and_fedot_pipelines/all/meta_features_preprocessors.pickle")
PIPELINES_FILE = get_data_dir() / "pymfe_meta_features_and_fedot_pipelines/all/pipelines_fedot.pickle"


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
    datasets = [OpenMLDataset.from_search(dset_name) for dset_name in datasets_names]
    # Load pipelines.
    with open(PIPELINES_FILE, "rb") as input_file:
        pipelines = pickle.load(input_file)
    # Make prediction.
    return advisor.predict(pipelines, datasets, k=5)


if __name__ == "__main__":
    result = main()
    print(result)
