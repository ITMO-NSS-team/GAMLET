import os.path

import pandas as pd
import yaml
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from typing import List

from experiments.fedot_warm_start.run import MF_EXTRACTOR_PARAMS, fetch_datasets
from meta_automl.data_preparation.file_system import get_project_root
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.meta_algorithm.datasets_similarity_assessors import KNeighborsBasedSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseFEDOTPipelineAdvisor


def get_contextual_bandit():
    repo = OperationTypesRepository.assign_repo('model', 'model_repository.json')
    mab = ContextualMultiArmedBanditAgent(actions=[parameter_change_mutation,
                                                   MutationTypesEnum.single_change,
                                                   MutationTypesEnum.single_drop,
                                                   MutationTypesEnum.single_add,
                                                   MutationTypesEnum.single_edge],
                                          context_agent_type=ContextAgentTypeEnum.nodes_num,
                                          available_operations=repo.operations)
    return mab


def gather_data_from_histories(path_to_dataset_similarity: str, datasets: List[str],
                               knowledge_base, datasets_dict):
    dataset_similarity = pd.read_csv(path_to_dataset_similarity)

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
                print(len(path_to_histories))


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
    print('a')
