from meta_automl.data_preparation.model_selectors import FEDOTPipelineSelector


def main():
    selector = FEDOTPipelineSelector(datasets=['cnae-9'], launch_dir=r'\\10.9.14.114\calc\Nikitin\aggregated_4h\agg')
    return selector.select()


if __name__ == '__main__':
    result = main()
