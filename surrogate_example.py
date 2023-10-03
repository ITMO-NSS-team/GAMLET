
from functools import partial

from fedot.api.main import Fedot
from fedot.core.repository.tasks import (Task, TaskTypesEnum,
                                         TsForecastingParams)
from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer

from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import OpenMLDatasetMetaFeaturesExtractor
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.surrogate.models import RankingPipelineDatasetSurrogateModel
from thegolem.data_pipeline_surrogate import DataPipelineSurrogate


if __name__ == '__main__':
    # Define data
    train_data = None # Specify your data here.
    open_ml_dataset_id = None # Specify your dataset OpenML ID here to get the dataset meta-features.

    # Create surrogate model
    pipeline_estimator = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
        checkpoint_path = "/home/cherniak/MetaFEDOT/experiments/openml_meta_features_and_fedot_pipelines/train_surrogate_model_best/hyperparameter_tuning/version_5/checkpoints/last.ckpt",
        hparams_file = "/home/cherniak/MetaFEDOT/experiments/openml_meta_features_and_fedot_pipelines/train_surrogate_model_best/hyperparameter_tuning/version_5/hparams.yaml"
    )

    pipeline_features_extractor = FEDOTPipelineFeaturesExtractor(include_operations_hyperparameters=False, operation_encoding="ordinal")

    features_preprocessor = FeaturesPreprocessor(load_path="/home/cherniak/MetaFEDOT/data/openml_meta_features_and_fedot_pipelines/all/meta_features_preprocessors.pickle")
    meta_features_extractor = OpenMLDatasetMetaFeaturesExtractor(features_preprocessors=features_preprocessor)
    dataset_meta_features = meta_features_extractor(dataset_id=open_ml_dataset_id)

    surrogate_model = DataPipelineSurrogate(
        pipeline_features_extractor=pipeline_features_extractor,
        dataset_meta_features=dataset_meta_features,
        pipeline_estimator=pipeline_estimator
    )

    # Create FEDOT. TODO: Change task from forecasting to classification.
    model = Fedot(
        problem='ts_forecasting',
        task_params=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=30)).task_params,
        timeout=None,
        n_jobs=-1,
        with_tuning=True,
        cv_folds=2,
        validation_blocks=2,
        preset='fast_train',
        optimizer=partial(SurrogateEachNgenOptimizer, surrogate_model=surrogate_model),
    )

    # Run AutoML model design as usual
    pipeline = model.fit(train_data)
