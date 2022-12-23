from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    loader = OpenMLDatasetsLoader()
    extractor = PymfeExtractor().fit({'groups': 'general'})

    cached_datasets = loader.load(dataset_names)
    meta_features = extractor.extract(cached_datasets)
    return meta_features


if __name__ == '__main__':
    result = main()
