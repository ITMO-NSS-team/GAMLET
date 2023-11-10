from meta_automl.data_preparation.datasets_loaders import TimeSeriesDatasetsLoader


def get_datasets():
    dataset_names = [
        'M4_D1002'
    ]
    datasets_loader = TimeSeriesDatasetsLoader(forecast_length=13)
    datasets = datasets_loader.load(dataset_names)
    print(f'Datasets "{", ".join(dataset_names)}" are downloaded.')
    return datasets


if __name__ == '__main__':
    result = get_datasets()
    print(1)
