from typing import Optional, List, Union, Callable

import openml
import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence
from openml import OpenMLTask, OpenMLFlow
from openml.tasks import TaskType

from meta_automl.data_preparation.dataset import DatasetCache
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetID
from meta_automl.data_preparation.model_selectors import ModelSelector


# TODO: Make it work.
class OpenMLSelector(ModelSelector):
    def __init__(self):
        self.datasets: Optional[List[OpenMLDatasetID]] = None
        self.selected_models: Optional[List[Union[Pipeline, List[Pipeline]]]] = None

    def fit(self, datasets: Optional[List[Union[DatasetCache, str]]] = None,
            task_type: TaskType = TaskType.SUPERVISED_CLASSIFICATION,
            select_best: bool = True, n_models: int = 1,
            task_suitability_filter: Callable[[OpenMLTask], bool] = lambda t: True,
            flow_suitability_filter: Callable[[OpenMLFlow], bool] = lambda f: True
            ):
        datasets = [d if isinstance(d, str) else d.name for d in datasets]
        datasets = [openml.datasets.get_dataset(d).dataset_id for d in datasets]
        df_datasets = openml.datasets.list_datasets(data_id=datasets, output_format="dataframe")
        tasks = {}
        for dataset_id in df_datasets["did"].values:
            tasks.update(openml.tasks.list_tasks(task_type=task_type, data_id=dataset_id))
        df_tasks = pd.DataFrame.from_dict(tasks, orient="index")
        df_evaluations = openml.evaluations.list_evaluations(tasks=list(df_tasks["tid"]), size=None,
                                                             function='area_under_roc_curve', output_format="dataframe")
        df_evaluations_sklearn = df_evaluations[df_evaluations['flow_name'].str.startswith('sklearn')]
        data_with_best_model = df_evaluations_sklearn.groupby(['tid', 'flow_name']).apply(
            lambda x: x.nlargest(1, ['value'])).reset_index(drop=True)

        self.datasets = datasets
        return self

    def select(self, n_best: int = 1, fit_from_scratch: bool = False):
        pipelines = []
        for n in ...:
            pipeline = ...
            if n_best > 1:
                ensure_wrapped_in_sequence(pipeline)
            pipelines.append(pipeline)
        self.selected_models = pipelines
        return pipelines
