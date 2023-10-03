# Uncomment the line below when GOLEM release will support meta optimizations:
# from golem.core.optimisers.meta.surrogate_model import SurrogateModel
# Currently replaced with the following:
# from .surrogate_optimization import SurrogateModel

"""
Example of use:

model = Fedot(problem='ts_forecasting',
    task_params=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon)).task_params,
    timeout=timeout,
    n_jobs=-1,
    with_tuning=with_tuning,
    cv_folds=2, validation_blocks=validation_blocks, preset='fast_train',
    optimizer=partial(SurrogateOptimizer, surrogate_model=SingleValueSurrogateModel()))

"""
from typing import Any

from .surrogate_optimization import SurrogateModel

# TODO: run example to understand what is `kwargs.get('objective').__self__._data_producer.args[0]`
# TODO: modify code to enable use of developed model
class SingleValueSurrogateModel(SurrogateModel):
    def __init__(self):
        self.model = model

    def __call__(self, graph, **kwargs: Any):
        # graph: GraphDelegate.

        #TODO: convert GraphDelegate to appropriate format
        # TODO: obtain dataset meta-features from input_data
        # TODO pass data to surrogate to obtain graph metric
        #

        # example how we can get input data from objective
        input_data = kwargs.get('objective').__self__._data_producer.args[0]
        return [0]
