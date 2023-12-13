import numpy as np
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.fitness import SingleObjFitness

from meta_automl.data_preparation.dataset import CustomDataset
from meta_automl.data_preparation.evaluated_model import EvaluatedModel
from meta_automl.data_preparation.model_fitness_scalers import DatasetModelsFitnessScaler


def test_dataset_models_scaler():
    fitness_metric_name = ['a', 'b', 'c']
    predictor = Pipeline(PipelineNode('rf'))
    datasets = [CustomDataset(i) for i in range(3)]
    dataset_ids = [dataset.id for dataset in datasets]
    n_models = 5
    models = [[EvaluatedModel(
        predictor=predictor, dataset=dataset,
        fitness=SingleObjFitness(dataset.id + i + 1, dataset.id + i + 2, dataset.id + i + 3),
        fitness_metric_name=fitness_metric_name) for i in range(n_models)] for dataset in datasets]
    scaler = DatasetModelsFitnessScaler().fit(dataset_ids, models)
    new_models_1 = scaler.transform(dataset_ids, models)
    new_models_2 = scaler.fit_transform(dataset_ids, models)

    assert np.array(new_models_1).shape == np.array(new_models_2).shape

    for d_m, d_m_1, d_m_2 in zip(models, new_models_1, new_models_2):
        for i, (m, m_1, m_2) in enumerate(zip(d_m, d_m_1, d_m_2)):
            assert m.fitness != m_1.fitness == m_2.fitness  # Initial fitness does not equal to scaled one.
            assert m_1.fitness.values == tuple([1 / (n_models - 1) * i] * len(fitness_metric_name))  # Expected values.
