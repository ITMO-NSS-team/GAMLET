import openml
import pandas as pd

from sklearn.model_selection import train_test_split
from typing import List

from meta_automl.data_preparation.dataset import OpenMLDatasetIDType


def openml_datasets_train_test_split(dataset_ids: List[OpenMLDatasetIDType],
                                     test_size: float, seed=None):
    df_openml_datasets = openml.datasets.list_datasets(dataset_ids, output_format='dataframe')
    df_openml_datasets_split_features = df_openml_datasets[
        ['name', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses']].copy(deep=False)
    for column in df_openml_datasets_split_features.columns[1:]:
        if column != 'NumberOfClasses':
            median = df_openml_datasets_split_features[column].median()
            df_openml_datasets_split_features[column] = \
                (df_openml_datasets_split_features[column] > median).map({False: 'small', True: 'big'})
        else:
            median = df_openml_datasets_split_features[column][df_openml_datasets_split_features[column] != 2].median()
            df_openml_datasets_split_features[column] = df_openml_datasets_split_features[column].apply(
                lambda n: 'binary' if n == 2 else {False: 'small', True: 'big'}[n > median])
    df_split_categories = df_openml_datasets_split_features.copy()
    df_split_categories['category'] = df_openml_datasets_split_features.apply(lambda row: '_'.join(
        row[1:]), axis=1)
    df_split_categories.drop(columns=['NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses'], inplace=True)
    # Group single-value categories into a separate category
    cat_counts = df_split_categories['category'].value_counts()
    single_value_categories = cat_counts[cat_counts == 1].index
    idx = df_split_categories[df_split_categories['category'].isin(single_value_categories)].index
    df_split_categories.loc[idx, 'category'] = 'single_value'
    df_datasets_to_split = df_split_categories[df_split_categories['category'] != 'single_value']
    df_test_only_datasets = df_split_categories[df_split_categories['category'] == 'single_value']
    if not df_datasets_to_split.empty:
        df_train_datasets, df_test_datasets = train_test_split(
            df_datasets_to_split,
            test_size=test_size,
            shuffle=True,
            stratify=df_datasets_to_split['category'],
            random_state=seed
        )
        # df_test_datasets = pd.concat([df_test_datasets, df_test_only_datasets])
    else:
        df_train_datasets, df_test_datasets = train_test_split(
            df_split_categories,
            test_size=test_size,
            shuffle=True,
            random_state=seed
        )
    df_train_datasets['is_train'] = 1
    df_test_datasets['is_train'] = 0
    df_split_datasets = df_test_datasets
    df_split_datasets = df_split_datasets.rename(columns={'name': 'dataset_name'})
    df_split_datasets.index.rename('dataset_id', inplace=True)

    return df_split_datasets


def main():
    dataset_ids = openml.study.get_suite(99).data
    df_split_datasets = openml_datasets_train_test_split(dataset_ids, test_size=0.3)
    df_split_datasets.to_csv('train_test_datasets_opencc18.csv')


if __name__ == '__main__':
    main()
