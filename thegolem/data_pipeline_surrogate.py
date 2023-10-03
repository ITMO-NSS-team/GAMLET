from typing import Any, Callable

import numpy as np
from fedot.core.pipelines.adapters import PipelineAdapter
from golem.core.dag.graph import Graph
from golem.core.optimisers.meta.surrogate_model import SurrogateModel


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
        self.dataset_meta_features = dataset_meta_features
        self.pipeline_estimator = pipeline_estimator
        self.pipeline_adapter = PipelineAdapter()

    def _graph2pipeline_string(self, graph: Graph) -> str:
        pipeline = self.pipeline_adapter._restore(graph)
        pipeline.unfit()
        pipline_string = pipeline.save()[0].encode()
        return pipline_string

    def __call__(self, graph: Graph, **kwargs: Any) -> float:
        pipline_string = self._graph2pipeline_string(graph)
        pipeline_features = self.pipeline_features_extractor(pipline_string)
        score = self.pipeline_estimator.predict(self.dataset_meta_features, pipeline_features)
        return score