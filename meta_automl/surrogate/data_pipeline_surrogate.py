from typing import Any, Callable, Dict

import json
import numpy as np
from fedot.core.pipelines.adapters import PipelineAdapter
from golem.core.dag.graph import Graph
from golem.core.optimisers.meta.surrogate_model import SurrogateModel

from torch_geometric.loader import DataLoader
import torch

def get_extractor_params(filename: str) -> Dict[str, str]:
    with open(filename) as f:
        extractor_params = json.load(f)
    return extractor_params

class DataPipelineSurrogate(SurrogateModel):
    """
    Surrogate model to evaluate FEDOT pipelines for given dataset.

    Parameters:
    ----------
    pipeline_features_extractor: Extractor of pipeline features.
    dataset_meta_features: Dataset meta-features. # TODO: subject of changes.
    pipeline_estimator: Pipeline estimator.

    Example of use:
    ---------------

    model = Fedot(problem='ts_forecasting',
        task_params=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon)).task_params,
        timeout=timeout,
        n_jobs=-1,
        with_tuning=with_tuning,
        cv_folds=2, validation_blocks=validation_blocks, preset='fast_train',
        optimizer=partial(SurrogateOptimizer, surrogate_model=SingleValueSurrogateModel()))

    """
    def __init__(
            self,
            pipeline_features_extractor: Callable,
            dataset_meta_features: np.ndarray,
            pipeline_estimator: Callable,
        ):
        self.pipeline_features_extractor = pipeline_features_extractor
        self.dataset_meta_features = torch.tensor(list(dataset_meta_features.values())).view(1,-1)
        self.pipeline_estimator = pipeline_estimator
        self.pipeline_estimator.eval()
        self.pipeline_adapter = PipelineAdapter()
        
    def _graph2pipeline_string(self, graph: Graph) -> str:
        pipeline = self.pipeline_adapter._restore(graph)
        pipeline.unfit()
        pipline_string = pipeline.save()[0].encode()
        return pipline_string

    def __call__(self, graph: Graph, **kwargs: Any) -> float:        
        pipline_string = self._graph2pipeline_string(graph)
        pipeline_features = self.pipeline_features_extractor(pipline_string)        
        pipeline_features.x = pipeline_features.x.view(-1)  # change if use model's hyperparameters!!!!
        if not pipeline_features.edge_index.shape[0]:
            pipeline_features.edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        
        loader = DataLoader([pipeline_features], batch_size=1)
        batch = next(iter(loader))
        
        with torch.no_grad():
            score = self.pipeline_estimator(batch, self.dataset_meta_features)
        score = score.view(-1).item()        
        return [score]

              
class PipelineVectorizer:              
    def __init__(
            self,
            pipeline_features_extractor: Callable,
            pipeline_estimator: Callable,
        ):
        self.pipeline_features_extractor = pipeline_features_extractor
        self.pipeline_estimator = pipeline_estimator
        self.pipeline_estimator.eval()
        self.pipeline_adapter = PipelineAdapter()

    def _graph2pipeline_string(self, graph: Graph) -> str:
        pipeline = self.pipeline_adapter._restore(graph)
        pipeline.unfit()
        pipline_string = pipeline.save()[0].encode()
        return pipline_string

    def _pipeline_to_batch(self, graph: Graph) -> Any:
        pipline_string = self._graph2pipeline_string(graph)
        pipeline_features = self.pipeline_features_extractor(pipline_string)        
        pipeline_features.x = pipeline_features.x.view(-1)  # change if use model's hyperparameters!!!!
        if not pipeline_features.edge_index.shape[0]:
            pipeline_features.edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        
        loader = DataLoader([pipeline_features], batch_size=1)
        batch = next(iter(loader))
        return batch
        
    def __call__(self, graph: Graph, **kwargs: Any) -> torch.Tensor:          
        batch = self._pipeline_to_batch(graph)     
        with torch.no_grad():
            return self.pipeline_estimator.pipeline_encoder(batch)
