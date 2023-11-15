from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from golem.core.optimisers.fitness import SingleObjFitness
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from fedot.core.pipelines.pipeline import Pipeline

from meta_automl.data_preparation.dataset import DatasetBase
from meta_automl.data_preparation.evaluated_model import EvaluatedModel
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.file_system.file_system import get_checkpoints_dir
from meta_automl.meta_algorithm.model_advisors import ModelAdvisor
from meta_automl.surrogate.data_pipeline_surrogate import get_extractor_params
from meta_automl.surrogate.surrogate_model import RankingPipelineDatasetSurrogateModel
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor

class SurrogateGNNModelAdvisor(ModelAdvisor):
    """Pipeline advisor based on surrogate GNN model.

    Parameters:
    -----------
    config:
        Dict of model parameters. The parameters are: TODO.
    """

    def __init__(
        self,
        meta_features_extractor: Optional[MetaFeaturesExtractor] = None,
        meta_features_preprocessor: Optional[FeaturesPreprocessor] = None,
pipeline_extractor: Optional[FEDOTPipelineFeaturesExtractor] = None,
    ):
        # loading surrogate model
        checkpoints_dir = get_checkpoints_dir() / 'tabular'
        self.surrogate_model = RankingPipelineDatasetSurrogateModel.load_from_checkpoint(
            checkpoint_path=checkpoints_dir / 'checkpoints/best.ckpt',
            hparams_file=checkpoints_dir / 'hparams.yaml'
        )
        self.surrogate_model.eval()

        self.meta_features_extractor = meta_features_extractor
        self.meta_features_preprocessor = meta_features_preprocessor
        self.pipeline_extractor = pipeline_extractor

    def _preprocess_dataset_features(self, dataset):
        x = self.meta_features_extractor.extract([dataset], fill_input_nans=True).fillna(0)
        x = self.meta_features_preprocessor.transform(x, single=False).fillna(0)
        transformed = x.groupby(by=['dataset', 'variable'])['value'].apply(list).apply(lambda x: pd.Series(x))
        dset_data = Data(torch.tensor(transformed.values, dtype=torch.float32))
        dset_data_loader = DataLoader([dset_data], batch_size=1)
        return next(iter(dset_data_loader))

    def _predict_single(
        self,
        dataset: DatasetBase,
        k: int,
        pipelines: List[Pipeline],
        pipeline_dataloader: DataLoader,
        ) -> List[EvaluatedModel]:
        """Select optimal pipelines for given dataset.
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
            for batch in pipeline_dataloader:
                scores.append(torch.squeeze(self.surrogate_model(batch, x_dset)).numpy())

        scores = np.array(scores)
        indx = np.argsort(scores)
        k = min(len(indx), k)
        best_models = []
        for i in indx[-k:][::-1]:
            best_models.append(
                EvaluatedModel(
                    pipelines[i],
                    SingleObjFitness(scores[i]),
                    'surrogate_fitness',
                    dataset))
        return best_models

    def predict(
        self,
        pipelines: List[Pipeline],
        datasets: List[DatasetBase],
        k: int = 5,
        pipelines_data: Optional[List[Data]] = None,
    ) -> List[List[EvaluatedModel]]:
        """Select optimal pipelines for given list of datasets.

        Parameters
        ----------
        datasets: List[DatasetBase]
            List of `DatasetBase` objects to select pipelines for.
        pipelines: List[Pipeline]
            List of pipelines to select from.
        k: int, optional
            Number of pipelines to predict per dataset (default: 5).
        pipelines_as_data: List[Data], optional
            Extracted pipelines data to infer surrogate on (default: None).

        Returns
        -------
        top_models : List[List[EvaluatedModel]]
            List of top models for each dataset.

        """
        if pipelines_data is not None:
            pipeline_dataloader = DataLoader(pipelines_data, batch_size=1)
        else:
            pipelines_data = [self.pipeline_extractor(pipeline.save()[0]) for pipeline in pipelines]
            pipeline_dataloader = DataLoader(pipelines_data, batch_size=1)
        return [self._predict_single(dset, k, pipelines, pipeline_dataloader) for dset in datasets]
