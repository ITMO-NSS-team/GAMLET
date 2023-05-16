import os

from golem.core.optimisers.adaptive.operator_agent import ExperienceBuffer
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory

from utils import project_root


def get_result_triplets(history_path: str, buffer: ExperienceBuffer):
    """ Get triplets from one history. """
    history = OptHistory().load(history_path)
    for gen in history.individuals:
        try:
            buffer.collect_results(gen.data)
        except IndexError:
            for ind in gen.data:
                try:
                    buffer.collect_result(ind)
                except IndexError:
                    print('Incorrect individual')


def get_all_triplets(path_to_datasets: str):
    """ Collect all triplets to one buffer. """
    datasets = os.listdir(path_to_datasets)
    buffer = ExperienceBuffer()
    for dataset in datasets:
        path_to_dataset = os.path.join(path_to_datasets, dataset, 'histories')
        launches = os.listdir(path_to_dataset)
        for launch in launches:
            path_to_histories = os.path.join(path_to_dataset, launch)
            histories = os.listdir(path_to_histories)
            for history in histories:
                path_to_history = os.path.join(path_to_histories, history)
                get_result_triplets(history_path=path_to_history, buffer=buffer)


if __name__ == '__main__':
    path_to_datasets = os.path.join(project_root(), 'data', 'knowledge_base_0', 'datasets')
    get_all_triplets(path_to_datasets=path_to_datasets)
