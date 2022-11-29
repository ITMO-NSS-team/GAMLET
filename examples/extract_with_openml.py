from components.data_preparation.features_extractor import OpenMLFeaturesExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    extractor = OpenMLFeaturesExtractor().fit()
    meta_features = extractor.extract(dataset_names)
    return meta_features


if __name__ == '__main__':
    result = main()
