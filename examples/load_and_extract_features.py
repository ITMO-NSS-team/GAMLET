from components.data_preparation.datasets_loader import OpenmlLoader
from components.data_preparation.features_extractor import PymfeExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    loader = OpenmlLoader(dataset_names)
    extractor = PymfeExtractor({'groups': 'general'})

    cached_datasets = loader.load()
    meta_features = extractor.extract(cached_datasets)
    return meta_features


if __name__ == '__main__':
    result = main()
