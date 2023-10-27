from functools import partial

import openml
from fedot.api.main import Fedot
from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.surrogate.data_pipeline_surrogate import DataPipelineSurrogate, get_extractor_params
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel

if __name__ == '__main__':
    dataset_name = 'sylvine'  # Specify your OpenML dataset here to get the dataset meta-features.
    datasets_loader = OpenMLDatasetsLoader()
    dset = openml.datasets.get_dataset(dataset_name)
    open_ml_dataset_id = dset.id
    train_data = datasets_loader.load_single(open_ml_dataset_id)
 
    # Load surrogate model
    surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
        checkpoint_path="./experiments/base/checkpoints/best.ckpt",
        hparams_file="./experiments/base/hparams.yaml"
    )
    surrogate_model.eval()
    
    # Prepare pipeline extractor
    pipeline_features_extractor = FEDOTPipelineFeaturesExtractor(include_operations_hyperparameters=False,
                                                                 operation_encoding="ordinal")   
    
    # Prepare dataset extractor and extract metafeatures
    extractor_params = get_extractor_params('configs/use_features.json')
    meta_features_extractor = PymfeExtractor(
        extractor_params = extractor_params,
    )
    meta_features_preprocessor = FeaturesPreprocessor(load_path= "./data/pymfe_meta_features_and_fedot_pipelines/all/meta_features_preprocessors.pickle",
                                                      extractor_params=extractor_params) 
    x_dset = meta_features_extractor.extract([train_data], fill_input_nans=True).fillna(0)
        
    # Compose extractors and model into joint sturcture
    surrogate_pipeline = DataPipelineSurrogate(
        pipeline_features_extractor=pipeline_features_extractor,
        dataset_meta_features=x_dset,
        meta_features_preprocessor = meta_features_preprocessor,
        pipeline_estimator=surrogate_model
    )
  
    # create FEDOT with SurrogateEachNgenOptimizer
    model = Fedot(
        problem='classification',
        timeout=5,
        n_jobs=-1,
        with_tuning=True,
        cv_folds=2,
        validation_blocks=2,
        preset='fast_train',
        optimizer=partial(SurrogateEachNgenOptimizer, surrogate_model=surrogate_pipeline, surrogate_each_n_gen=1),
    )
    
    # Run AutoML model design as usual
    dat = train_data.get_data()
    pipeline = model.fit(features=dat.x, target=dat.y)
