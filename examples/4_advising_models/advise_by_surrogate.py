""" Example on advising pipelines for a dataset. """

import pickle

import pandas as pd
import torch
from torch_geometric.data import Data

from meta_automl.data_preparation.dataset import DatasetBase, OpenMLDataset
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.file_system.file_system import get_checkpoints_dir, get_configs_dir, get_data_dir
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor, PymfeExtractor
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.meta_algorithm.model_advisors import SurrogateGNNModelAdvisor
from meta_automl.surrogate.data_pipeline_surrogate import get_extractor_params
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel

SURROGATE_MODEL_CHECKPOINT_FILE = get_checkpoints_dir() / "tabular" / "checkpoints" / "best.ckpt"
SURROGATE_MODEL_HYPERPARAMETERS_FILE = get_checkpoints_dir() / "tabular" / "hparams.yaml"
META_FEATURES_EXTRACTOR_CONFIG_FILE = get_configs_dir() / "use_features.json"
META_FEATURES_PREPROCESSOR_FILE = (
        get_data_dir() / "pymfe_meta_features_and_fedot_pipelines/all/meta_features_preprocessors.pickle"
)
PIPELINES_FILE = get_data_dir() / "pymfe_meta_features_and_fedot_pipelines/all/pipelines_fedot.pickle"


def preprocess_dataset_features(
        dataset: DatasetBase,
        dataset_meta_features_extractor: MetaFeaturesExtractor,
        dataset_meta_features_preprocessor: FeaturesPreprocessor,
) -> Data:
    """Extract dataset features.

    Parameters
    ----------
    dataset: Dataset to extract features from.

    dataset_meta_features_extractor: Extractor returning dataset meta-features.

    dataset_meta_features_preprocessor: Preprocessor, preparing the meta-features after extraction.

    Returns
    -------
    dset_data : Data
        Dataset features.

    """
    x = dataset_meta_features_extractor.extract([dataset], fill_input_nans=True, use_cached=False).fillna(0)
    x = dataset_meta_features_preprocessor.transform(x, single=False).fillna(0)
    transformed = x.groupby(by=["dataset", "variable"])["value"].apply(list).apply(lambda x: pd.Series(x))
    dset_data = Data(x=torch.tensor(transformed.values, dtype=torch.float32))
    return dset_data


def main():
    # Build model to estimate pipelines.
    surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
        checkpoint_path=SURROGATE_MODEL_CHECKPOINT_FILE,
        hparams_file=SURROGATE_MODEL_HYPERPARAMETERS_FILE,
    )
    # Build dataset meta features extractor.
    extractor_params = get_extractor_params(META_FEATURES_EXTRACTOR_CONFIG_FILE)
    dataset_meta_features_extractor = PymfeExtractor(**extractor_params)
    # Build dataset meta features preprocessor.
    dataset_meta_features_preprocessor = FeaturesPreprocessor(load_path=META_FEATURES_PREPROCESSOR_FILE)
    # Build pipeline features extractor.
    pipeline_extractor = FEDOTPipelineFeaturesExtractor(
        include_operations_hyperparameters=False,
        operation_encoding="ordinal",
    )
    # Build adviser.
    advisor = SurrogateGNNModelAdvisor(surrogate_model)
    # Load datasets.
    datasets_names = [
        "apsfailure",
    ]
    datasets = [OpenMLDataset.from_search(dset_name) for dset_name in datasets_names]
    # Load pipelines.
    with open(PIPELINES_FILE, "rb") as input_file:
        pipelines = pickle.load(input_file)

    # Extract features
    pipelines_features = [pipeline_extractor(pipeline.save()[0]) for pipeline in pipelines]
    datasets_features = [
        preprocess_dataset_features(dataset, dataset_meta_features_extractor, dataset_meta_features_preprocessor) for
        dataset in datasets]

    # Make prediction.
    return advisor.predict(pipelines, datasets, pipelines_features, datasets_features, k=5)


if __name__ == "__main__":
    result = main()
    print(result)
