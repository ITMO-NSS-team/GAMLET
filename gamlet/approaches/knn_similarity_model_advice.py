from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from sklearn.preprocessing import MinMaxScaler

from gamlet.approaches import MetaLearningApproach
from gamlet.components.dataset_similarity_assessors import KNeighborsSimilarityAssessor
from gamlet.components.meta_features_extractors import DatasetMetaFeatures, PymfeExtractor
from gamlet.components.model_advisors import DiverseModelAdvisor
from gamlet.components.model_fitness_scalers import DatasetModelsFitnessScaler, ScalerType
from gamlet.components.models_loaders import FedotHistoryLoader
from gamlet.data_preparation.dataset import DatasetIDType, OpenMLDataset, TabularData
from gamlet.data_preparation.evaluated_model import EvaluatedModel


class KNNSimilarityModelAdvice(MetaLearningApproach):
    def __init__(self, n_best_dataset_models_to_memorize: int,
                 mf_extractor_params: dict, assessor_params: dict, advisor_params: dict):
        self.parameters = self.Parameters(
            n_best_dataset_models_to_memorize=n_best_dataset_models_to_memorize,
            mf_extractor_params=mf_extractor_params,
            assessor_params=assessor_params,
            advisor_params=advisor_params,
        )
        self.components = self.Components(
            models_loader=FedotHistoryLoader(),
            models_fitness_scaler=DatasetModelsFitnessScaler(MinMaxScaler),
            mf_extractor=PymfeExtractor(**mf_extractor_params),
            mf_scaler=MinMaxScaler(),
            datasets_similarity_assessor=KNeighborsSimilarityAssessor(**assessor_params),
            model_advisor=DiverseModelAdvisor(**advisor_params),
        )
        self.data = self.Data()

    @dataclass
    class Parameters:
        n_best_dataset_models_to_memorize: int
        mf_extractor_params: dict = field(default_factory=dict)
        assessor_params: dict = field(default_factory=dict)
        advisor_params: dict = field(default_factory=dict)

    @dataclass
    class Components:
        models_loader: FedotHistoryLoader
        models_fitness_scaler: DatasetModelsFitnessScaler
        mf_extractor: PymfeExtractor
        mf_scaler: ScalerType
        datasets_similarity_assessor: KNeighborsSimilarityAssessor
        model_advisor: DiverseModelAdvisor

    @dataclass
    class Data:
        meta_features: DatasetMetaFeatures = None
        datasets: List[OpenMLDataset] = None
        datasets_data: List[OpenMLDataset] = None
        dataset_ids: List[DatasetIDType] = None
        best_models: List[List[EvaluatedModel]] = None

    def fit(self,
            datasets_data: Sequence[TabularData],
            histories: Sequence[Sequence[OptHistory]],
            evaluate_model_func: Optional[Sequence[Callable]] = None):
        data = self.data
        params = self.parameters

        data.datasets_data = list(datasets_data)
        data.datasets = [d.dataset for d in datasets_data]
        data.dataset_ids = [d.id for d in datasets_data]

        data.meta_features = self.extract_train_meta_features(data.datasets_data)
        self.fit_datasets_similarity_assessor(data.meta_features, data.dataset_ids)

        data.best_models = self.load_models(data.datasets, histories, params.n_best_dataset_models_to_memorize,
                                            evaluate_model_func)
        self.fit_model_advisor(data.dataset_ids, data.best_models)

        return self

    def load_models(
            self, datasets: Sequence[OpenMLDataset],
            histories: Sequence[Sequence[OptHistory]],
            n_best_dataset_models_to_load: int,
            evaluate_model_func: Optional[Sequence[Callable]] = None) -> Sequence[Sequence[EvaluatedModel]]:
        models = self.components.models_loader.load(datasets, histories, n_best_dataset_models_to_load,
                                                    evaluate_model_func)
        models = self.components.models_fitness_scaler.fit_transform(models, datasets)
        return models

    def extract_train_meta_features(self, datasets_data: List[TabularData]) -> DatasetMetaFeatures:
        components = self.components

        meta_features = components.mf_extractor.extract(
            datasets_data, fill_input_nans=True)

        meta_features.fillna(0, inplace=True)

        meta_features[meta_features.columns] = components.mf_scaler.fit_transform(meta_features)

        return meta_features

    def fit_datasets_similarity_assessor(self, meta_features: DatasetMetaFeatures, dataset_ids: List[DatasetIDType]
                                         ) -> KNeighborsSimilarityAssessor:
        return self.components.datasets_similarity_assessor.fit(meta_features, dataset_ids)

    def fit_model_advisor(self, dataset_ids: List[DatasetIDType], best_models: Sequence[Sequence[EvaluatedModel]]
                          ) -> DiverseModelAdvisor:
        return self.components.model_advisor.fit(dataset_ids, best_models)

    def predict(self, datasets_data: Sequence[TabularData]) -> List[List[EvaluatedModel]]:
        mf_extractor = self.components.mf_extractor
        mf_scaler = self.components.mf_scaler
        assessor = self.components.datasets_similarity_assessor
        advisor = self.components.model_advisor

        meta_features = mf_extractor.extract(datasets_data, fill_input_nans=True)
        meta_features.fillna(0, inplace=True)
        meta_features[meta_features.columns] = mf_scaler.transform(meta_features)
        similar_dataset_ids = assessor.predict(meta_features)
        models = advisor.predict(similar_dataset_ids)

        return models
