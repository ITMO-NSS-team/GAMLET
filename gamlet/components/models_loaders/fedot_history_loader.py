from itertools import chain
from typing import Callable, List, Optional, Sequence

from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.repository.default_params_repository import DefaultOperationParamsRepository
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory

from gamlet.data_preparation.dataset import DatasetBase
from gamlet.data_preparation.evaluated_model import EvaluatedModel
from gamlet.components.models_loaders import ModelsLoader


def extract_best_models_from_history(
        dataset: DatasetBase,
        history: OptHistory,
        n_best_models_to_load: int,
        evaluate_model_func: Optional[Callable] = None,
) -> List[EvaluatedModel]:
    best_individuals_accum = []
    generations = history.generations
    if generations:
        best_individuals = sorted(chain(*generations),
                                  key=lambda ind: ind.fitness,
                                  reverse=True)
        for individual in history.final_choices or []:
            if individual not in best_individuals:
                best_individuals.insert(0, individual)

        best_individuals = list({ind.graph.descriptive_id: ind for ind in best_individuals}.values())
        best_individuals = best_individuals[:n_best_models_to_load - 1]

        node_params_repo = DefaultOperationParamsRepository()
        for individual in best_individuals:
            pipeline = PipelineAdapter().restore(individual.graph)
            for node in pipeline.nodes:
                node.parameters = node_params_repo.get_default_params_for_operation(node.name)
            if evaluate_model_func:
                fitness, metric_names = evaluate_model_func(pipeline)
            else:
                fitness = individual.fitness or SingleObjFitness()
                metric_names = history.objective.metric_names
            model = EvaluatedModel(pipeline, fitness, metric_names, dataset)
            best_individuals_accum.append(model)

    return best_individuals_accum


class FedotHistoryLoader(ModelsLoader):

    def load(self,
             datasets: Sequence[DatasetBase],
             histories: Sequence[Sequence[OptHistory]],
             n_best_dataset_models_to_load: int,
             evaluate_model_func: Optional[Sequence[Callable]] = None,
             ) -> List[List[EvaluatedModel]]:
        result = []
        if evaluate_model_func is not None:
            for dataset, histories, eval_func in zip(datasets, histories, evaluate_model_func):
                result += [
                    extract_best_models_from_history(dataset, history, n_best_dataset_models_to_load, eval_func)
                    for history in histories]
        else:
            for dataset, histories in zip(datasets, histories):
                result += [
                    extract_best_models_from_history(dataset, history, n_best_dataset_models_to_load)
                    for history in histories]
        return result
