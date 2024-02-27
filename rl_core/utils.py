import os

from sklearn.model_selection import train_test_split

from meta_automl.utils import project_root
from rl_core.dataloader import DataLoader_TS


def define_data_for_experiment():
    data_folder_path = os.path.join(str(project_root()), 'MetaFEDOT\\data\\knowledge_base_time_series_0\\datasets\\')
    dataset_names = [name for name in os.listdir(data_folder_path)]

    train, test = train_test_split(dataset_names, test_size=3)

    train_datasets = {}
    for dataset in train:
        train_datasets[dataset] = os.path.join(data_folder_path, f'{dataset}/data.csv')

    test_datasets = {}
    for dataset in test:
        test_datasets[dataset] = os.path.join(data_folder_path, f'{dataset}/data.csv')

    path_to_meta_data = os.path.join(str(project_root()),
                                     'MetaFEDOT\\data\\knowledge_base_time_series_0\\meta_features_ts.csv')

    dataloader = DataLoader_TS(train_datasets, path_to_meta_data=path_to_meta_data)

    return dataloader, train, test
