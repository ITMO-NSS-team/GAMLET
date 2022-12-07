from meta_automl.data_preparation.model_selectors import FEDOTResultsBestPipelineSelector


def main():
    selector = FEDOTResultsBestPipelineSelector().fit(datasets=['DATASET_NAME'],
                                                      launch_dir=r'LAUNCHES_PATH')
    return selector.select()


if __name__ == '__main__':
    result = main()
