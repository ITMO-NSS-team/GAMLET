from dataclasses import dataclass

import numpy as np
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory

from gamlet.approaches import MetaLearningApproach
from gamlet.approaches.knn_similarity_model_advice import KNNSimilarityModelAdvice
from gamlet.data_preparation.dataset import OpenMLDataset
from gamlet.data_preparation.file_system import get_data_dir
from test.constants import OPENML_CACHED_DATASETS


class LinearTransformer:
    def __init__(self, k: float, b: float):
        self.k = k
        self.b = b

    def transform(self, x):
        return self.k * x + self.b


class TestMetaLearningApproach(MetaLearningApproach):
    @dataclass
    class Parameters:
        linear_k: float
        linear_b: float

    @dataclass
    class Components:
        linear_transformer: LinearTransformer

    @dataclass
    class Data:
        x = None

    def __init__(self, linear_k, linear_b):
        self.parameters = self.Parameters(linear_k=linear_k, linear_b=linear_b)
        self.components = self.Components(linear_transformer=LinearTransformer(k=linear_k, b=linear_b))
        self.data = self.Data()

    def predict(self, x: np.ndarray):
        self.data.x = x
        return self.components.linear_transformer.transform(x)


def test_meta_learning_approach():
    k, b = 2, 1
    approach = TestMetaLearningApproach(k, b)
    x = np.array([1, 2, 3])
    x_pred = approach.predict(x)
    assert np.allclose(x * k + b, x_pred)


def test_knn_similarity_model_advice():
    dataset_ids_train, dataset_ids_test = OPENML_CACHED_DATASETS, [OPENML_CACHED_DATASETS[0]]
    datasets_train, datasets_test = ([OpenMLDataset(id_).get_data() for id_ in dataset_ids] for dataset_ids in
                                     (dataset_ids_train, dataset_ids_test))
    history_path = get_data_dir() / 'test_history.json'
    history = OptHistory.load(history_path)
    histories_train = [[history]] * 2

    approach = KNNSimilarityModelAdvice(n_best_dataset_models_to_memorize=1, mf_extractor_params={},
                                        assessor_params={}, advisor_params=dict(n_best_to_advise=1))
    approach.fit(datasets_train, histories_train)
    prediction = approach.predict(datasets_test)

    assert np.array(prediction).shape == (1, 1)
    assert prediction[0][0].predictor == history.tuning_result
