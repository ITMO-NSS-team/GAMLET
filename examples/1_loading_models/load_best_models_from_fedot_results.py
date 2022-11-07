from meta_automl.data_preparation.models_loaders import FEDOTPipelinesLoader


def main():
    models_loader = FEDOTPipelinesLoader(datasets_to_load=['cnae-9'], launch_dir=r'\\10.9.14.114\calc\Nikitin\aggregated_4h\agg')
    return models_loader.load()


if __name__ == '__main__':
    result = main()
