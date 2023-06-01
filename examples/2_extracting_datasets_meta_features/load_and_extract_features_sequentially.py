import openml

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    dataset_ids = [openml.datasets.get_dataset(name, download_data=False).dataset_id for name in dataset_names]
    loader = OpenMLDatasetsLoader()
    extractor = PymfeExtractor(extractor_params={'groups': 'general'})

    datasets = loader.load(dataset_ids)
    meta_features = extractor.extract(datasets)
    return meta_features


if __name__ == '__main__':
    result = main()
