from components.data_preparation.datasets_loader import OpenMLDatasetsLoader
from components.data_preparation.features_extractor import PymfeExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    extractor = PymfeExtractor().fit({'groups': 'general'})
    meta_features = extractor.extract(dataset_names)
    return meta_features


if __name__ == '__main__':
    result = main()
