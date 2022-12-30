from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader


def get_datasets():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    datasets_loader = OpenMLDatasetsLoader()
    datasets = datasets_loader.load(dataset_names)
    print(f'Datasets "{", ".join(dataset_names)}" are available at the paths:')
    print('\n'.join(str(d) for d in datasets))
    return datasets


if __name__ == '__main__':
    result = get_datasets()
