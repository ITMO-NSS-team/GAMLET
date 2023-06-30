import openml

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine'
    ]
    dataset_ids = [openml.datasets.get_dataset(name, download_data=False, download_qualities=False).dataset_id for name in dataset_names]
    extractor = PymfeExtractor(extractor_params={'groups': 'general'}, datasets_loader=OpenMLDatasetsLoader())
    meta_features = extractor.extract(dataset_ids)
    return meta_features


if __name__ == '__main__':
    result = main()
