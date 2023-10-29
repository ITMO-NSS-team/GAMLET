"""Before running this script you should have collected pipelines with different hyperparameters on different datasets.
This script unpacks the collected data and provide numerical indices for each pipeline and dataset stored in files.

WARNING: This script does not save it state. So, if the script was stopped, you need to re-run it from scratch!
"""
import atexit
import os
import pickle
from pathlib import Path

import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.tuning.tuner_interface import BaseTuner

# Uncomment this section if you want to use pipeline extractor.
# from meta_automl.data_preparation.pipeline_features_extractors import (
#     FEDOTPipelineFeaturesExtractor,
#     JoinedAutoencoderHyperparametersEmbedder,
# )
from tqdm import tqdm


def exit_handler(dataset_directory, records, id2pipeline_path, id2dataset_id):
    with open(Path(dataset_directory).joinpath("records.pickle"), "wb") as f:
        pickle.dump(records, f)
    with open(Path(dataset_directory).joinpath("id2pipeline_path.pickle"), "wb") as f:
        pickle.dump(id2pipeline_path, f)
    with open(Path(dataset_directory).joinpath("id2dataset_id.pickle"), "wb") as f:
        pickle.dump(id2dataset_id, f)

def main(dataset_directory, split_path, hyperparameters_directory, pipelines_directory, pipeline_extractor=None):
    available_datasets = os.listdir(hyperparameters_directory)
    split = pd.read_csv(split_path)
    split = split[split.dataset_id.apply(lambda x: x in available_datasets)]
    assert len(split) > split.is_train.sum()  # check that there is at least single test dataset

    pipeline_counter = 0
    dataset_counter = 0
    id2pipeline_path = {}
    id2dataset_id = {}

    records = []

    # To not loose computed data on early exit
    atexit.register(exit_handler, dataset_directory=dataset_directory, records=records, id2pipeline_path=id2pipeline_path, id2dataset_id=id2dataset_id)

    for dataset_id in tqdm(split.dataset_id):
        id2dataset_id[dataset_counter] = dataset_id

        split_name = "train" if split[split.dataset_id == dataset_id].is_train.values[0] else "test"
        hyperparams_files = list(Path(hyperparameters_directory).joinpath(dataset_id).glob("*.pickle"))
        for hyperparams_file in tqdm(hyperparams_files):
            with open(hyperparams_file, "rb") as f:
                optuna_study = pickle.load(f)
            pipeline_name = hyperparams_file.name.replace(".pickle", ".json")
            pipeline_file = Path(pipelines_directory).joinpath(dataset_id).joinpath("models").joinpath(pipeline_name)
            for i, trial in enumerate(optuna_study.trials):
                metric = trial.value
                parameters = trial.params
                pipeline = Pipeline().load(pipeline_file)
                pipeline = BaseTuner.set_arg_graph(pipeline, parameters)

                # Decide whether data is transformed with pre-trained encoders.
                if pipeline_extractor is not None:
                    pipeline_json_string, _ = pipeline.save()
                    data = pipeline_extractor(pipeline_json_string)
                else:
                    data, _ = pipeline.save()

                data_file = (
                    Path(dataset_directory)
                    .joinpath(split_name)
                    .joinpath(dataset_id)
                    .joinpath(hyperparams_file.name.replace(".pickle", ""))
                    .joinpath(f"{i}.pickle")
                )
                if not data_file.parent.exists():
                    data_file.parent.mkdir(parents=True)
                with open(data_file, "wb") as f:
                    pickle.dump(data, f)
                id2pipeline_path[pipeline_counter] = data_file
                records.append({"task_id": dataset_counter, "pipeline_id": pipeline_counter, "metric": metric})
                pipeline_counter += 1
        dataset_counter += 1

    with open(Path(dataset_directory).joinpath("records.pickle"), "wb") as f:
        pickle.dump(records, f)
    with open(Path(dataset_directory).joinpath("id2pipeline_path.pickle"), "wb") as f:
        pickle.dump(id2pipeline_path, f)
    with open(Path(dataset_directory).joinpath("id2dataset_id.pickle"), "wb") as f:
        pickle.dump(id2dataset_id, f)

    # If everything is done correctly unregister exit handler.
    atexit.unregister(exit_handler)


if __name__ == "__main__":
    # DEFINE YOUR PATHS HERE.
    dataset_directory = "/home/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines_raw"
    split_path = "/home/cherniak/itmo_job/GAMLET/data/knowledge_base_1_v2/train_test_datasets_classification.csv"
    hyperparameters_directory = "/home/cherniak/itmo_job/graphs_with_hyperparameters"
    pipelines_directory = "/home/cherniak/itmo_job/GAMLET/data/knowledge_base_1_v2/datasets"

    # Uncomment this section if you want to use pipeline extractor.
    # hyperparameters_embedder = JoinedAutoencoderHyperparametersEmbedder(
    #     ckpt_path="/home/cherniak/itmo_job/GAMLET/experiment_logs/embed_hyperparameters/to_4_with_learnables/checkpoints/epoch=9-step=980.ckpt",
    #     embedding_dim=4,
    #     with_learnable=True,
    # )

    # pipeline_extractor = FEDOTPipelineFeaturesExtractor(
    #     operation_encoding=None,
    #     hyperparameters_embedder=hyperparameters_embedder,
    # )

    # Comment this line if you want to use pipeline extractor.
    pipeline_extractor = None

    main(dataset_directory, split_path, hyperparameters_directory, pipelines_directory, pipeline_extractor)
