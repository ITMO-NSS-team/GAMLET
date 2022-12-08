from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine'
    ]
    extractor = PymfeExtractor().fit({'groups': 'general'}, datasets_loader=OpenMLDatasetsLoader())
    meta_features = extractor.extract(dataset_names)
    return meta_features


if __name__ == '__main__':
    result = main()
