from gamlet.components.models_loaders import KnowledgeBaseModelsLoader


if __name__ == '__main__':
    models_loader = KnowledgeBaseModelsLoader()

    train_datasets = models_loader.parse_datasets('train')  # DataFrame in the form (dataset_id, dataset_name)
    # test_datasets = models_loader.parse_datasets('test')  # DataFrame in the form (dataset_id, dataset_name)
    # all_datasets = models_loader.parse_datasets()  # DataFrame in the form (dataset_id, dataset_name, is_train)

    # Load train/test/all models
    # train_models = models_loader.load(train_datasets['dataset_name'])
    # test_models = models_loader.load(train_datasets['dataset_name'])
    # all_models = models_loader.load()

    # ===== Another way to get train models, but also group them by datasets:
    models_for_train = {}

    for dataset_id in train_datasets['dataset_id']:
        dataset_models = models_loader.load(
            dataset_ids=[dataset_id],   # load models just for this exact dataset.
            fitness_metric='logloss',       # must correspond to a metric name in a knowledge base.
        )
        models_for_train[dataset_id] = dataset_models

        # If you need to load data to the local storage
        # dataset = OpenMLDatasetsLoader().load_single(dataset_name)

        # If you need meta-features for the dataset
        # extractor_params = {...}
        # meta-features = PymfeExtractor(extractor_params).extract([dataset_name])
