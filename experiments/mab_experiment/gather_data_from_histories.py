import os.path

import pandas as pd
import yaml
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.operator_agent import ExperienceBuffer
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from typing import List

from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from tqdm import tqdm

from experiments.fedot_warm_start.run import MF_EXTRACTOR_PARAMS, fetch_datasets
from experiments.mab_experiment.run import _load_pipeline_vectorizer
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.meta_algorithm.datasets_similarity_assessors import KNeighborsBasedSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseFEDOTPipelineAdvisor


def get_contextual_bandit():
    repo = OperationTypesRepository.assign_repo('model', 'model_repository.json')
    context_agent_type = _load_pipeline_vectorizer()
    mab = ContextualMultiArmedBanditAgent(actions=[parameter_change_mutation,
                                                   MutationTypesEnum.single_change,
                                                   MutationTypesEnum.single_drop,
                                                   MutationTypesEnum.single_add,
                                                   MutationTypesEnum.single_edge],
                                          context_agent_type=context_agent_type,
                                          available_operations=repo.operations)
    return mab


def gather_data_from_histories(path_to_dataset_similarity: str, datasets: List[str],
                               knowledge_base, datasets_dict):
    dataset_similarity = pd.read_csv(path_to_dataset_similarity)

    mab_per_dataset = dict.fromkeys(datasets, None)

    for original_dataset_name in datasets:
        similar = dataset_similarity[dataset_similarity['dataset'] == original_dataset_name]['similar_datasets'].tolist()[0]\
            .replace("[", "").replace("]", "").split(" ")

        for s in similar:
            if s != '':
                n = int(s)
                dataset_name = datasets_dict[n].name
                if dataset_name == original_dataset_name or dataset_name not in list(set(knowledge_base['dataset_name'].tolist())):
                    continue
                path_to_histories = list(set(list(knowledge_base[knowledge_base['dataset_name'] == dataset_name]['history_path'])))
                if len(path_to_histories) == 0:
                    continue
                mab_per_dataset[original_dataset_name] = \
                    train_bandit_on_histories(path_to_histories, get_contextual_bandit())


def train_bandit_on_histories(path_to_histories: List[str], bandit: ContextualMultiArmedBanditAgent):
    """ Retrieve data from histories and train bandit on it. """
    experience_buffer = ExperienceBuffer()
    for path_to_history in tqdm(path_to_histories):
        history = OptHistory.load(path_to_history)
        for i, gen in enumerate(history.individuals):
            if i == 10:
                break
            individuals = gen.data
            for ind in individuals:
                # simplify and use only the first operator and parent
                if not ind.parent_operator or \
                        not ind.parent_operator.operators or not ind.parent_operator.parent_individuals:
                    continue

                try:
                    operator = ind.parent_operator.operators[0]
                    # get mutation class since it is stored as str
                    operator = _get_mutation_class(operator)
                    parent_individual = ind.parent_operator.parent_individuals[0]
                    fitness_difference = ind.fitness.value - parent_individual.fitness.value
                    if ind.graph and operator and fitness_difference:
                        experience_buffer.collect_experience(obs=ind.graph, action=operator, reward=fitness_difference)
                # since some individuals have None fitness
                except TypeError:
                    continue

            # check if the is any experience collected
            if experience_buffer._observations:
                bandit.partial_fit(experience=experience_buffer)
    return bandit


def _get_mutation_class(mutation_str: str):
    if 'parameter_change_mutation' in mutation_str:
        return parameter_change_mutation
    if 'MutationTypesEnum' in mutation_str:
        return MutationTypesEnum[mutation_str.split('.')[1]]


def get_similar_datasets(path_to_save: str, datasets_dict) \
        -> pd.DataFrame:

    dataset_ids = list(datasets_dict.keys())
    datasets = list(datasets_dict.values())

    # Meta Features
    extractor = PymfeExtractor(extractor_params=MF_EXTRACTOR_PARAMS)
    meta_features = extractor.extract(dataset_ids, fill_input_nans=True)
    meta_features = meta_features.fillna(0)
    # Datasets similarity
    data_similarity_assessor = KNeighborsBasedSimilarityAssessor(
        n_neighbors=min(len(dataset_ids), 5))
    data_similarity_assessor.fit(meta_features, dataset_ids)

    df = pd.DataFrame({'dataset_id': [], 'dataset': [], 'similar_datasets': []})

    for dataset_id, dataset in zip(dataset_ids, datasets):
        cur_meta_features = extractor.extract([dataset_id], fill_input_nans=True)
        cur_meta_features = cur_meta_features.fillna(0)
        try:
            similar_datasets = data_similarity_assessor.predict(cur_meta_features)
            row = {'dataset_id': dataset_id, 'dataset': dataset.name, 'similar_datasets': similar_datasets}
            df = df.append(row, ignore_index=True)
        except ValueError:
            continue

    df.to_csv(path_to_save)
    print(f'Dataset similarity table was saved to {path_to_save}')
    return df


if __name__ == '__main__':
    base_path = os.path.join(get_project_root(), 'experiments',
                             'mab_experiment')
    dataset_similaruty_path = os.path.join(base_path, 'dataset_similarity.csv')

    df_datasets_train, df_datasets_test, datasets_dict = fetch_datasets()
    # get_similar_datasets(base_path, datasets_dict)

    path_to_knowledge_base = os.path.join(base_path, 'knowledge_base.csv')
    knowledge_base = pd.read_csv(path_to_knowledge_base)

    config_name = 'mab_config.yaml'
    path_to_config = os.path.join(get_project_root(), 'experiments', 'mab_experiment', config_name)
    with open(path_to_config, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
    datasets = config_dict['datasets']

    gather_data_from_histories(path_to_dataset_similarity=dataset_similaruty_path,
                               datasets=datasets,
                               knowledge_base=knowledge_base,
                               datasets_dict=datasets_dict)
