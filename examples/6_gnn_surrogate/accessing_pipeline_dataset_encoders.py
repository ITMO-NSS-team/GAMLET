from gamlet.components.datasets_loaders import OpenMLDatasetsLoader
from gamlet.components.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from gamlet.data_preparation.file_system.file_system import get_checkpoints_dir
from gamlet.surrogate.data_pipeline_surrogate import PipelineVectorizer
from gamlet.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel

if __name__ == '__main__':
    dataset_name = 'higgs'
    datasets_loader = OpenMLDatasetsLoader()
    dataset = datasets_loader.load_single(dataset_name, allow_name=True)
    checkpoints_dir = get_checkpoints_dir() / 'tabular'
    # Load surrogate model
    surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
        checkpoint_path=checkpoints_dir / 'checkpoints/best.ckpt',
        hparams_file=checkpoints_dir / 'hparams.yaml'
    )
    surrogate_model.eval()

    # Trained pipeline encoder model (pipeline -> vec)
    print(surrogate_model.pipeline_encoder)

    # Trained dataset encoder model (dataset -> vec)
    print(surrogate_model.dataset_encoder)

    pipeline_features_extractor = FEDOTPipelineFeaturesExtractor(include_operations_hyperparameters=False,
                                                                 operation_encoding='ordinal')
    pipeline_vectorizer = PipelineVectorizer(
        pipeline_features_extractor=pipeline_features_extractor,
        pipeline_estimator=surrogate_model
    )
    # vector = pipeline_vectorizer(pipeline)
