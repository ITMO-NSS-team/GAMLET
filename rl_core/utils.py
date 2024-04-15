import os
import pandas as pd

from sklearn.model_selection import train_test_split

from meta_automl.utils import project_root
from rl_core.dataloader import TimeSeriesDataLoader

OFFLINE_TRAJECTORIES = [
    [2, 0],
    [9, 0],
    [11, 0],
    [12, 0],
    [13, 0],
    [25, 1, 31, 0],
    [25, 3, 31, 0],
    [25, 4, 31, 0],
    [25, 5, 31, 0],
    [25, 6, 31, 0],
    [25, 7, 31, 0],
    [25, 8, 31, 0],
    [25, 10, 31, 0],
    [25, 9, 31, 0],
    [25, 10, 31, 0],
    [25, 14, 31, 0],
]


def define_data_for_experiment(test_size: int = 3):
    data_folder_path = os.path.join(str(project_root()), 'MetaFEDOT\\data\\knowledge_base_time_series_0\\datasets\\')
    dataset_names = [name for name in os.listdir(data_folder_path)]

    temp = pd.read_csv('pipeline_validation_results.csv', index_col=0)
    train = temp[temp['Topo Pipeline'].isna() == True]['Dataset'].to_list()
    test = temp[temp['Topo Pipeline'].isna() == False]['Dataset'].to_list()

    # if test_size:
    #     train, test = train_test_split(dataset_names, test_size=3)
    # else:
    #     train, test = dataset_names, []

    train_datasets = {}
    for dataset in train:
        train_datasets[dataset] = os.path.join(data_folder_path, f'{dataset}/data.csv')

    test_datasets = {}
    for dataset in test:
        test_datasets[dataset] = os.path.join(data_folder_path, f'{dataset}/data.csv')

    path_to_meta_data = os.path.join(str(project_root()),
                                     'MetaFEDOT\\data\\knowledge_base_time_series_0\\meta_features_ts.csv')

    dataloader_train = TimeSeriesDataLoader(train_datasets, path_to_meta_data=path_to_meta_data)
    dataloader_test = TimeSeriesDataLoader(test_datasets, path_to_meta_data=path_to_meta_data)

    return dataloader_train, dataloader_test, train, test,
