"""Script to be refactored. Current data is single metric only (check script to collect dataset with hyperparameters)"""
from meta_automl.data_preparation.datasets_loaders.custom_datasets_loader import CustomDatasetsLoader
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.surrogate_dataset.knowledge_base_to_dataset import KnowledgeBaseToDataset, \
    dataset_from_id_without_data_loading
from meta_automl.surrogate.data_pipeline_surrogate import get_extractor_params
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.data_preparation.pipeline_features_extractors import JoinedAutoencoderHyperparametersEmbedder
import os
import pandas as pd
import optuna
from pathlib import Path
import pickle
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.tuning.tuner_interface import BaseTuner
from tqdm import tqdm



dataset_directory="/Users/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines_raw"
split_path = "/Users/cherniak/itmo_job/GAMLET/data/knowledge_base_1_v2/train_test_datasets_classification.csv"
hyperparameters_directory = "/Users/cherniak/itmo_job/graphs_with_hyperparameters"
pipelines_directory = "/Users/cherniak/itmo_job/GAMLET/data/knowledge_base_1_v2/datasets"

available_datasets = os.listdir(hyperparameters_directory)
split = pd.read_csv(split_path)
split = split[split.dataset_id.apply(lambda x: x in available_datasets)]
assert len(split) > split.is_train.sum()  # check that there is at least single test dataset

pipeline_counter = 0
dataset_counter = 0
id2pipeline_path = {}
id2dataset_id = {}

records = []

for dataset_id in tqdm(split.dataset_id):
    id2dataset_id[dataset_counter] = dataset_id
    
    split_name = "train" if split[split.dataset_id == dataset_id].is_train.values[0] else "test"
    hyperparams_files = list(Path(hyperparameters_directory).joinpath(dataset_id).glob("*.pickle"))
    for hyperparams_file in hyperparams_files:
        with open(hyperparams_file, "rb") as f:
            optuna_study = pickle.load(f)
        pipeline_name = hyperparams_file.name.replace(".pickle", ".json")
        pipeline_file = Path(pipelines_directory).joinpath(dataset_id).joinpath("models").joinpath(pipeline_name)
        for i, trial in enumerate(optuna_study.trials):
            metric = trial.value
            parameters = trial.params
            pipeline = Pipeline().load(pipeline_file)
            pipeline = BaseTuner.set_arg_graph(pipeline, parameters)
            data, _ = pipeline.save()
            data_file = Path(dataset_directory) \
                        .joinpath(split_name) \
                        .joinpath(dataset_id) \
                        .joinpath(hyperparams_file.name.replace(".pickle", "")) \
                        .joinpath(f"{i}.pickle")
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
 
