from gamlet.components.datasets_loaders import OpenMLDatasetsLoader
from gamlet.components.meta_features_extractors import PymfeExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    loader = OpenMLDatasetsLoader()
    extractor = PymfeExtractor(groups='general')

    datasets = loader.load(dataset_names, allow_names=True)
    meta_features = extractor.extract(datasets)
    return meta_features


if __name__ == '__main__':
    result = main()
    print(result)
