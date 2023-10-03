from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import OpenMLDatasetMetaFeaturesExtractor
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.model_advisors import ModelAdvisor
from meta_automl.surrogate import models
from meta_automl.surrogate.training import get_pipelines_dataset


class SurrogateGNNPipelineAdvisor(ModelAdvisor):
    """Pipeline advisor based on surrogate GNN model.

    Parameters:
    -----------
    config:
        Dict of model parameters. The parameters are: TODO.
    """

    def __init__(self, config: Dict[str, Any]):
        # loading surrogate model
        model_class = getattr(models, config["model"].pop("name"))
        self.surrogate_model = model_class.load_from_checkpoint(
            checkpoint_path=config["model_data"]["save_dir"] + "checkpoints/last.ckpt",
            hparams_file=config["model_data"]["save_dir"] + "hparams.yaml"
        )
        self.surrogate_model.eval()

        # loading pipelines data
        data_ppls, self.pipelines = get_pipelines_dataset(config["pipelines_data"]["root_path"])
        self.loader = DataLoader(data_ppls, batch_size=config["batch_size"])

        # loading dataset metafeature processor
        features_preprocessor = FeaturesPreprocessor(
            load_path=config["pipelines_data"]["root_path"] + "meta_features_preprocessors.pickle",
        )
        self.meta_features_extractor = OpenMLDatasetMetaFeaturesExtractor(features_preprocessors=features_preprocessor)

    def predict(self, dataset: pd.DataFrame, k: int = 5) -> List[List[Model]]:
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

        open_ml_dataset_id = 11
        dataset_meta_features = self.meta_features_extractor(dataset_id=open_ml_dataset_id)
        # TODO: Rewrite after making work `FeaturesExtractor`
        # accepting DataFrame, not `dataset_id`

        dataset_meta_features = torch.tensor(list(dataset_meta_features.values())).view(1, -1)
        scores = []
        with torch.no_grad():
            for batch in self.loader:
                scores.append(torch.squeeze(self.surrogate_model(batch, dataset_meta_features)).numpy())

        scores = np.array(scores)
        indx = np.argsort(scores)
        k = min(len(indx), k)
        return [self.pipelines[i] for i in indx[-k:][::-1]], [scores[i] for i in indx[-k:][::-1]]
