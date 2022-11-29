from components.data_preparation.datasets_loader import OpenMLDatasetsLoader


def get_datasets():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    loader = OpenMLDatasetsLoader(dataset_names)
    datasets = loader.load()
    print(f'Datasets "{", ".join(dataset_names)}" are available at the paths:')
    print('\n'.join(str(d) for d in datasets))
    return datasets


if __name__ == '__main__':
    result = get_datasets()
