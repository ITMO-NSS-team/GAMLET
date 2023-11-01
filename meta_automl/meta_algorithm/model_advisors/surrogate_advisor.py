from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from golem.core.optimisers.fitness import SingleObjFitness
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from meta_automl.data_preparation.dataset import DatasetBase
from meta_automl.data_preparation.evaluated_model import EvaluatedModel
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.file_system import get_data_dir
from meta_automl.data_preparation.file_system.file_system import get_configs_dir
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.meta_algorithm.model_advisors import ModelAdvisor
from meta_automl.surrogate.data_pipeline_surrogate import get_extractor_params
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel


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
        surrogate_knowledge_base_dir = get_data_dir() / 'knowledge_base_surrogate'
        self.surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
            checkpoint_path=surrogate_knowledge_base_dir / 'checkpoints/best.ckpt',
            hparams_file=surrogate_knowledge_base_dir / 'hparams.yaml'
        )
        self.surrogate_model.eval()

        # Prepare dataset extractor and extract metafeatures
        config_dir = get_configs_dir() / 'use_features.json'
        extractor_params = get_extractor_params(config_dir)
        self.meta_features_extractor = PymfeExtractor(
            extractor_params=extractor_params,
        )
        self.meta_features_preprocessor = FeaturesPreprocessor(
            load_path=get_data_dir() / "pymfe_meta_features_and_fedot_pipelines/all/meta_features_preprocessors.pickle",
            extractor_params=extractor_params)

    def _preprocess_dataset_features(self, dataset):
        x = self.meta_features_extractor.extract([dataset], fill_input_nans=True, use_cached=False).fillna(0)
        x = self.meta_features_preprocessor.transform(x, single=False).fillna(0)
        transformed = x.groupby(by=['dataset', 'variable'])['value'].apply(list).apply(lambda x: pd.Series(x))
        dset_data = Data()
        dset_data.x = torch.tensor(transformed.values, dtype=torch.float32)
        dset_data_loader = DataLoader([dset_data], batch_size=1)
        return next(iter(dset_data_loader))

    def _predict_single(self, dataset: DatasetBase, k) -> List[EvaluatedModel]:
        """Predict optimal pipelines for given dataset.
        Parameters
        ----------
        dataset: DatasetBase
            dataset
        k: int, optional
            number of pipelines to predict (default: 5)
        Returns
        -------
        top_models : [Model]
            Top models for dataset.
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
                EvaluatedModel(
                    self.pipelines_fedot[i],
                    SingleObjFitness(scores[i]),
                    'surrogate_fitness',
                    dataset))
        return best_models

    def predict(self, datasets: List[DatasetBase], k: int = 5) -> List[List[EvaluatedModel]]:
        """Predict optimal pipelines for given list of datasets dataset.
        Parameters
        ----------
        datasets: List[DatasetBase]
            list of DatasetBase objects
        k: int, optional
            number of pipelines to predict (default: 5)
        Returns
        -------
        top_models : List[List[EvaluatedModel]]
            List of top models for each dataset.
        """
        return [self._predict_single(dset, k) for dset in datasets]
