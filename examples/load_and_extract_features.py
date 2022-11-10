from components.datasets_loader import OpenmlLoader
from components.features_extractor import PymfeExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    loader = OpenmlLoader(dataset_names)
    extractor = PymfeExtractor('general')

    datasets = loader()
    meta_features = extractor(datasets)
    return meta_features


if __name__ == '__main__':
    result = main()
