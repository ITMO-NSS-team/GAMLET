import sys,os
sys.path.append(os.getcwd())
from functools import partial

from fedot.api.main import Fedot
from fedot.core.repository.tasks import (Task, TaskTypesEnum,
                                         TsForecastingParams)
from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer
from thegolem.data_pipeline_surrogate import DataPipelineSurrogate

from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import OpenMLDatasetMetaFeaturesExtractor
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.surrogate.models import RankingPipelineDatasetSurrogateModel
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader

import openml

if __name__ == '__main__':
    # Define data
    
    dataset_name = 'higgs' # Specify your OpenML dataset here to get the dataset meta-features.
    datasets_loader = OpenMLDatasetsLoader()
    dset = openml.datasets.get_dataset(dataset_name)
    open_ml_dataset_id = dset.id
    train_data = datasets_loader.load([dataset_name])
    # train_data = dset.get_data()[0]  #datasets_loader.load(dataset_name) # Specify your data here.
    
   
    
    # Create surrogate model
    pipeline_estimator = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
        checkpoint_path = "./experiments/base/checkpoints/last.ckpt",
        hparams_file = "./experiments/base/hparams.yaml"
    )

    pipeline_features_extractor = FEDOTPipelineFeaturesExtractor(include_operations_hyperparameters=False, operation_encoding="ordinal")

    features_preprocessor = FeaturesPreprocessor(load_path="./data/openml_meta_features_and_fedot_pipelines/all/meta_features_preprocessors.pickle")
    meta_features_extractor = OpenMLDatasetMetaFeaturesExtractor(features_preprocessors=features_preprocessor)
    dataset_meta_features = meta_features_extractor(dataset_id=open_ml_dataset_id)

    surrogate_model = DataPipelineSurrogate(
        pipeline_features_extractor=pipeline_features_extractor,
        dataset_meta_features=dataset_meta_features,
        pipeline_estimator=pipeline_estimator
    )
    
    model = Fedot(
        problem='classification',
        timeout=5,
        n_jobs=-1,
        with_tuning=True,
        cv_folds=2,
        validation_blocks=2,
        preset='best_quality',
        optimizer=partial(SurrogateEachNgenOptimizer, surrogate_model=surrogate_model),
    )
    print('*******************************************************************************************')
    # Run AutoML model design as usual
    print(type(train_data[0]))
    
    pipeline = model.fit(train_data[0])
