import os
import sys

sys.path.append(os.getcwd())

import openml

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.surrogate.data_pipeline_surrogate import PipelineVectorizer
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel

if __name__ == '__main__':
    dataset_name = 'higgs'
    datasets_loader = OpenMLDatasetsLoader()
    dset = openml.datasets.get_dataset(dataset_name)
    open_ml_dataset_id = dset.id
    train_data = datasets_loader.load_single(open_ml_dataset_id)

    # Load surrogate model
    surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
        checkpoint_path="./experiments/base/checkpoints/last.ckpt",
        hparams_file="./experiments/base/hparams.yaml"
    )
    surrogate_model.eval()

    # Trained pipeline encoder model (pipeline -> vec)
    print(surrogate_model.pipeline_encoder)

    # Trained dataset encoder model (dataset -> vec)
    print(surrogate_model.dataset_encoder)

    pipeline_features_extractor = FEDOTPipelineFeaturesExtractor(include_operations_hyperparameters=False,
                                                                 operation_encoding="ordinal")
    pipeline_vectorizer = PipelineVectorizer(
        pipeline_features_extractor=pipeline_features_extractor,
        pipeline_estimator=surrogate_model
    )
    # vector = pipeline_vectorizer(pipeline)
