from meta_automl.data_preparation.meta_features_extractors import OpenMLFeaturesExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    extractor = OpenMLFeaturesExtractor()
    meta_features = extractor.extract(dataset_names)
    return meta_features


if __name__ == '__main__':
    result = main()
