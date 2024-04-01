import numpy as np
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.fitness import SingleObjFitness

from gamlet.components.model_fitness_scalers import DatasetModelsFitnessScaler
from gamlet.data_preparation.dataset import CustomDataset
from gamlet.data_preparation.evaluated_model import EvaluatedModel


def test_dataset_models_scaler():
    fitness_metric_names = ['a', 'b', 'c']
    predictor = Pipeline(PipelineNode('rf'))
    datasets = [CustomDataset(i) for i in range(3)]
    n_models = 5
    models = [
        [
            EvaluatedModel(
                predictor=predictor, dataset=dataset,
                metrics={
                    fitness_metric_names[0]: SingleObjFitness(dataset.id + i + 1),
                    fitness_metric_names[1]: SingleObjFitness(dataset.id + i + 2),
                    fitness_metric_names[2]: SingleObjFitness(dataset.id + i + 3)
                }
            )
            for i in range(n_models)]
        for dataset in datasets]
    scaler = DatasetModelsFitnessScaler().fit(models, datasets)
    new_models_1 = scaler.transform(models, datasets)
    new_models_2 = scaler.fit_transform(models, datasets)

    assert np.array(new_models_1).shape == np.array(new_models_2).shape

    for d_m, d_m_1, d_m_2 in zip(models, new_models_1, new_models_2):
        for i, (m, m_1, m_2) in enumerate(zip(d_m, d_m_1, d_m_2)):
            expected_value = 1 / (n_models - 1) * i
            for metric_name in fitness_metric_names:
                # Initial fitness does not equal to scaled one.
                assert m.metrics[metric_name] != m_1.metrics[metric_name] == m_2.metrics[metric_name]
                assert m_1.metrics[metric_name] == expected_value
