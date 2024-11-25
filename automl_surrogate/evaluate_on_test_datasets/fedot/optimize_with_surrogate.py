"""This module contains FEDOT evaluation code and required changes to use a surrogate model."""

import os
import pickle
import time
import timeit
from datetime import datetime
from functools import partial
from typing import Optional

import fedot
import numpy as np
import pandas as pd
import torch
import yaml
from fedot.api.main import Fedot
from fedot.core.pipelines.adapters import PipelineAdapter
from golem.core.adapter import BaseOptimizationAdapter
from golem.core.optimisers.genetic.evaluation import DelegateEvaluator
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.meta.surrogate_evaluator import SurrogateDispatcher as SurrogateDispatcher_
from golem.core.optimisers.meta.surrogate_model import RandomValuesSurrogateModel, SurrogateModel
from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer as SurrogateEachNgenOptimizer_
from golem.core.optimisers.objective.objective import GraphFunction, to_fitness
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError, _try_unfit_graph

from automl_surrogate.data.data_types import HeterogeneousBatch
from automl_surrogate.data.fedot_pipeline_features_extractor import FEDOTPipelineFeaturesExtractor2
from automl_surrogate.models import FusionRankNet

if fedot.__version__ == "0.7.2":
    from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
else:
    from fedot.core.repository.metrics_repository import ClassificationMetricsEnum


class SurrogateDispatcher(SurrogateDispatcher_):
    """Evaluates objective function with surrogate model.
    Usage: call `dispatch(objective_function)` to get evaluation function.
    Additionally, we need to pass surrogate_model object
    """

    def __init__(
        self,
        adapter: BaseOptimizationAdapter,
        n_jobs: int = 1,
        graph_cleanup_fn: Optional[GraphFunction] = None,
        delegate_evaluator: Optional[DelegateEvaluator] = None,
        surrogate_model: SurrogateModel = RandomValuesSurrogateModel(),
    ):
        super().__init__(adapter, n_jobs, graph_cleanup_fn, delegate_evaluator)
        self._n_jobs = 1
        self.surrogate_model = surrogate_model

    def evaluate_population(self, individuals: "PopulationT") -> "Optional[PopulationT]":
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(individuals)
        graphs = [ind.graph for ind in individuals_to_evaluate]
        uids = [ind.uid for ind in individuals_to_evaluate]
        evaluation_results = self.evaluate_batch(graphs, uids)
        # evaluation_results = [self.evaluate_single(ind.graph, ind.uid) for ind in individuals_to_evaluate]
        individuals_evaluated = self.apply_evaluation_results(individuals_to_evaluate, evaluation_results)
        evaluated_population = individuals_evaluated + individuals_to_skip or None
        return evaluated_population

    def evaluate_batch(self, graphs: "List[OptGraph]", uids: "List[str]") -> "List[GraphEvalResult]":
        start_time = timeit.default_timer()
        predictions = self.surrogate_model(graphs, objective=self._objective_eval)
        fitnesses = [to_fitness(p) for p in predictions]
        end_time = timeit.default_timer()
        eval_reses = []
        for fitness, uid_of_individual, graph in zip(fitnesses, uids, graphs):
            eval_res = GraphEvalResult(
                uid_of_individual=uid_of_individual,
                fitness=fitness,
                graph=graph,
                metadata={
                    "computation_time_in_seconds": end_time - start_time,
                    "evaluation_time_iso": datetime.now().isoformat(),
                    "surrogate_evaluation": True,
                },
            )
            eval_reses.append(eval_res)
        return eval_reses


class SurrogateOptimizer(SurrogateEachNgenOptimizer_):
    def __init__(
        self,
        objective: "Objective",
        initial_graphs: "Sequence[OptGraph]",
        requirements: "GraphRequirements",
        graph_generation_params: "GraphGenerationParams",
        graph_optimizer_params: "GPAlgorithmParameters",
        surrogate_model=RandomValuesSurrogateModel(),
    ):
        super().__init__(
            objective,
            initial_graphs,
            requirements,
            graph_generation_params,
            graph_optimizer_params,
            surrogate_model,
            surrogate_each_n_gen=1,
        )
        self.surrogate_dispatcher = SurrogateDispatcher(
            adapter=graph_generation_params.adapter,
            n_jobs=requirements.n_jobs,
            graph_cleanup_fn=_try_unfit_graph,
            delegate_evaluator=graph_generation_params.remote_evaluator,
            surrogate_model=surrogate_model,
        )

    def optimise(self, objective: "ObjectiveFunction") -> "Sequence[OptGraph]":
        surrogate_evaluator = self.surrogate_dispatcher.dispatch(objective, self.timer)

        with self.timer, self._progressbar:
            self._initial_population(surrogate_evaluator)
            while not self.stop_optimization():
                try:
                    new_population = self._evolve_population(surrogate_evaluator)
                except EvaluationAttemptsError as ex:
                    self.log.warning(f"Composition process was stopped due to: {ex}")
                    return [ind.graph for ind in self.best_individuals]
                # Adding of new population to history
                self._update_population(new_population)
        self._update_population(self.best_individuals, "final_choices")
        return [ind.graph for ind in self.best_individuals]


class SurrogatePipeline:
    """Performs required step to produce scores for given pipelines and meta-dataset.
    Current implementaion requires for meta-features to be computed in advance. Meta-features scaling is done within the module.
    Current implementation works with RankNet-based surrogate model.
    """

    def __init__(
        self,
        precomputed_meta_features_file: str,
        ckpt_root: str,
        ckpt_name: str,
    ):
        self.adapter = PipelineAdapter()
        self.pipe_ext = FEDOTPipelineFeaturesExtractor2(operation_encoding="ordinal")

        meta_features_scaler_path = os.path.join(ckpt_root, "scaler.pickle")
        with open(meta_features_scaler_path, "rb") as f:
            meta_features_scaler = pickle.load(f)

        datasets_meta_feats_all = pd.read_csv(precomputed_meta_features_file, index_col=0)
        self.datasets_meta_feats_all = pd.DataFrame(
            data=meta_features_scaler.transform(datasets_meta_feats_all.to_numpy()),
            index=datasets_meta_feats_all.index,
            columns=datasets_meta_feats_all.columns,
        )

        config_path = os.path.join(ckpt_root, "config.yml")
        ckpt_path = os.path.join(ckpt_root, "checkpoints", f"{ckpt_name}.ckpt")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = FusionRankNet(**{k: v for k, v in config["model"].items() if k != "class"})
        model.load_state_dict(ckpt["state_dict"])
        model = model.eval()
        self.model = model

    def set_dataset(self, dataset_id: str):
        """Set a dataset meta-features vector to be used during inference."""
        dataset_meta_feats = self.datasets_meta_feats_all.loc[dataset_id].to_numpy().reshape(1, -1)
        self.dataset_id = dataset_id
        self.dataset_meta_feats = torch.FloatTensor(dataset_meta_feats)

    def __call__(self, graphs: "List[OptGraph]", *args, **kwargs) -> "List[float]":
        """Given a list of graphs, predict scores on how good each graph is."""
        jsons_strs = [self.adapter._restore(g).save()[0] for g in graphs]
        pipes = [self.pipe_ext(g) for g in jsons_strs]
        batch = [HeterogeneousBatch.from_heterogeneous_data_list([p]) for p in pipes]
        with torch.no_grad():
            scores = self.model.forward(batch, self.dataset_meta_feats)
        return scores.reshape(-1, 1).numpy().tolist()


def create_and_test_pipeline(
    logger,
    dataset_name: str,
    x: "np.ndarray",
    y: "np.ndarray",
    x_test: "np.ndarray",
    y_test: "np.ndarray",
    surrogate: "SurrogatePipeline" = None,
    predefined_model: "Pipeline" = None,
):
    if surrogate is not None:
        surrogate.set_dataset(dataset_name)

    print("Creating FEDOT")
    kwargs = dict(
        timeout=360,  # Non-default
        n_jobs=1,  # Non-default
        available_operations=[  # Non-default
            "bernb",
            "dt",
            "fast_ica",
            "isolation_forest_class",
            "knn",
            "lgbm",
            "logit",
            "mlp",
            "normalization",
            "pca",
            "poly_features",
            "qda",
            "resample",
            "rf",
            "scaling",
        ],
        parallelization_mode="sequential",  # Non-default
        problem="classification",
        metric=(ClassificationMetricsEnum.ROCAUC, ClassificationMetricsEnum.logloss),
        with_tuning=False,
        preset="best_quality",  # Non-default
        logging_level=20,
    )
    if surrogate is not None:
        kwargs["optimizer"] = partial(SurrogateOptimizer, surrogate_model=surrogate)

    model = Fedot(**kwargs, use_pipelines_cache=False, use_preprocessing_cache=False)
    logger.info(f"Fitting FEDOT")
    t1 = time.time()
    if predefined_model is not None:
        _ = model.fit(features=x, target=y, predefined_model=predefined_model)
    else:
        _ = model.fit(features=x, target=y)
    elapsed = time.time() - t1
    logger.info(f"Elapsed time {elapsed}")
    logger.info(f"Predicting FEDOT")
    model.predict(x_test)
    logger.info(f"Prediction on test: {model.get_metrics(target=y_test)}")
    logger.info("")
