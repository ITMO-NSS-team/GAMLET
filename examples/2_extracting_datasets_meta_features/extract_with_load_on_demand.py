from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor


def main():
    dataset_names = [
        'nomao', 'sylvine'
    ]
    datasets = OpenMLDatasetsLoader(allow_names=True).load(dataset_names)
    extractor = PymfeExtractor(extractor_params={'groups': 'general'})
    meta_features = extractor.extract(datasets)
    return meta_features


if __name__ == '__main__':
    result = main()
    print(result)
