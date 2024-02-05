from gamlet.components.datasets_loaders import OpenMLDatasetsLoader


def get_datasets():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    datasets_loader = OpenMLDatasetsLoader()
    datasets = datasets_loader.load(dataset_names, allow_names=True)
    print(f'Datasets "{", ".join(dataset_names)}" are downloaded.')
    return datasets


if __name__ == '__main__':
    result = get_datasets()
