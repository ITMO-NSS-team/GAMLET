import openml.datasets

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader


def get_datasets():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    dataset_ids = [openml.datasets.get_dataset(name, download_data=False).dataset_id for name in dataset_names]
    datasets_loader = OpenMLDatasetsLoader()
    datasets = datasets_loader.load(dataset_ids)
    print(f'Datasets "{", ".join(dataset_names)}" are downloaded.')
    return datasets


if __name__ == '__main__':
    result = get_datasets()
