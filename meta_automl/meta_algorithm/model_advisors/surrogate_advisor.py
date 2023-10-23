from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.model_advisors import ModelAdvisor
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel
from meta_automl.data_preparation.dataset import DatasetBase
from meta_automl.surrogate.data_pipeline_surrogate import get_extractor_params
from golem.core.optimisers.fitness import SingleObjFitness

class SurrogateGNNPipelineAdvisor(ModelAdvisor):
    """Pipeline advisor based on surrogate GNN model.

    Parameters:
    -----------
    config:
        Dict of model parameters. The parameters are: TODO.
    """

    def __init__(self, config: Dict[str, Any], pipelines, pipelines_fedot):
        pipelines = pipelines
        self.pipelines_fedot = pipelines_fedot
        self.pipeline_dataloader = DataLoader(pipelines, batch_size=1)
        
        # loading surrogate model
        self.surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
            checkpoint_path="./experiments/base/checkpoints/best.ckpt",
            hparams_file="./experiments/base/hparams.yaml"
        )
        self.surrogate_model.eval()

        # Prepare dataset extractor and extract metafeatures
        extractor_params = get_extractor_params('configs/use_features.json')
        self.meta_features_extractor = PymfeExtractor(
            extractor_params = extractor_params,
        )
        self.meta_features_preprocessor = FeaturesPreprocessor(load_path= "./data/pymfe_meta_features_and_fedot_pipelines/all/meta_features_preprocessors.pickle",
                                                          extractor_params=extractor_params) 

    def _preprocess_dataset_features(self, dataset):
        x = self.meta_features_extractor.extract([dataset], fill_input_nans=True).fillna(0)
        x = self.meta_features_preprocessor.transform(x, single=False).fillna(0)
        transformed = x.groupby(by=['dataset', 'variable'])['value'].apply(list).apply(lambda x: pd.Series(x))     
        dset_data = Data()
        dset_data.x = torch.tensor(transformed.values, dtype=torch.float32)
        dset_data_loader = DataLoader([dset_data], batch_size=1)
        return next(iter(dset_data_loader))

    def predict(self, dataset: DatasetBase, k: int = 5) -> List[Model]:
        """Predict optimal pipelines for given dataset.
        Parameters
        ----------
        dataset: pd.DataFrame
            pandas DataFrame of a dataset
        k: int, optional
            number of pipelines to predict (default: 5)
        Returns
        -------
        top_pipelines : [Pipeline]
            Top pipelines for dataset.
        scores : [float]
            Scores of the returned pipelines.
        """     
        x_dset = self._preprocess_dataset_features(dataset)
        scores = []
        with torch.no_grad():
            for batch in self.pipeline_dataloader:
                scores.append(torch.squeeze(self.surrogate_model(batch, x_dset)).numpy())
                
        scores = np.array(scores)
        indx = np.argsort(scores)
        k = min(len(indx), k)    
        best_models = []
        for i in indx[-k:][::-1]:
            best_models.append(
                Model(
                    self.pipelines_fedot[i], 
                    SingleObjFitness(scores[i]), 
                    'surrogate_fitness', 
                    None))
        return best_models
