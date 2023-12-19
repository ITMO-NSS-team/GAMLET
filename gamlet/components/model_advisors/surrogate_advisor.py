from typing import List

import numpy as np
import torch
import torch.nn as nn
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.fitness import SingleObjFitness
from torch_geometric.data import Batch, Data

from gamlet.data_preparation.dataset import DatasetBase
from gamlet.data_preparation.evaluated_model import EvaluatedModel
from gamlet.components.model_advisors import ModelAdvisor


class SurrogateGNNModelAdvisor(ModelAdvisor):
    """ModelAdvisor based on dataset-and-pipeline-features-aware Graph Neural Network.

    Parameters:
    -----------
    surrogate_model: nn.Module
        Surrogate model to be used.
    """

    def __init__(
        self,
        surrogate_model: nn.Module,
    ):
        self.surrogate_model = surrogate_model
        self.surrogate_model.eval()
        self.device = next(self.surrogate_model.parameters()).device

    def _predict_single(
        self,
        dataset: DatasetBase,
        dataset_features: Data,
        pipelines: List[Pipeline],
        pipelines_features: List[Data],
        k: int,
    ) -> List[EvaluatedModel]:
        """Select optimal pipelines for given dataset.

        Parameters
        ----------
        dataset: DatasetBase
            Dataset to select pipelines for.
        dataset_features: Data
            Dataset meta-features.
        pipelines: List[Pipeline]
            List of pipelines to select from.
        pipelines_features: List[Data]
            Extracted pipelines features to infer surrogate on.
        k: int, optional
            Number of pipelines to predict (default: 5)

        Returns
        -------
        top_models : [Model]
            Top models for dataset.

        """
        dataset_features = Batch.from_data_list([dataset_features]).to(self.device)

        scores = []
        with torch.no_grad():
            for pipeline_features in pipelines_features:
                pipeline_features = Batch.from_data_list([pipeline_features]).to(self.device)
                score = self.surrogate_model(pipeline_features, dataset_features).cpu().item()
                scores.append(score)

        scores = np.array(scores)
        indx = np.argsort(scores)
        k = min(len(indx), k)
        best_models = []
        for i in indx[-k:][::-1]:
            best_models.append(EvaluatedModel(pipelines[i], SingleObjFitness(scores[i]), "surrogate_fitness", dataset))
        return best_models

    def predict(
        self,
        pipelines: List[Pipeline],
        datasets: List[DatasetBase],
        pipelines_features: List[Data],
        datasets_features: List[Data],
        k: int = 5,
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
        pipelines_features: List[Data], optional
            Extracted pipelines features to infer surrogate on (default: None).
        datasets_features: List[Data], optional
            Extracted datasets features to infer surrogate on (default: None).

        Returns
        -------
        top_models : List[List[EvaluatedModel]]
            List of top models for each dataset.

        """

        top_models = []
        for dset, dset_feats in zip(datasets, datasets_features):
            top_models.append(self._predict_single(dset, dset_feats, pipelines, pipelines_features, k))
        return top_models
