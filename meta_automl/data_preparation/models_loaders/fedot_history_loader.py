from itertools import chain
from typing import Dict, List

from fedot.core.pipelines.adapters import PipelineAdapter
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory

from meta_automl.data_preparation.dataset import DatasetBase, DatasetIDType
from meta_automl.data_preparation.model import Model
from meta_automl.data_preparation.models_loaders import ModelsLoader


def extract_best_models_from_history(dataset: DatasetBase, history: OptHistory,
                                     n_best_models_to_load: int) -> List[Model]:
    best_models = []
    if history.individuals:
        best_individuals = sorted(chain(*history.individuals),
                                  key=lambda ind: ind.fitness,
                                  reverse=True)
        for individual in history.final_choices or []:
            if individual not in best_individuals:
                best_individuals.insert(0, individual)

        best_individuals = list({ind.graph.descriptive_id: ind for ind in best_individuals}.values())
        best_individuals = best_individuals[:n_best_models_to_load - 1]

        for individual in best_individuals:
            pipeline = PipelineAdapter().restore(individual.graph)
            fitness = individual.fitness or SingleObjFitness()
            model = Model(pipeline, fitness, history.objective.metric_names[0], dataset)
            best_models.append(model)

    if history.tuning_result:
        final_pipeline = PipelineAdapter().restore(history.tuning_result)
        final_model = Model(final_pipeline, SingleObjFitness(), history.objective.metric_names[0], dataset)
        best_models.insert(0, final_model)

    return best_models


class FedotHistoryLoader(ModelsLoader):

    def load(self, datasets, histories, n_best_dataset_models_to_load: int) -> List[List[Model]]:
        result = [extract_best_models_from_history(dataset, history, n_best_dataset_models_to_load)
                  for dataset, history in zip(datasets, histories)]

        return result
