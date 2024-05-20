from __future__ import annotations

import json
import logging
import os
import sys
import pickle
import shutil
import timeit
from datetime import datetime, timedelta
from functools import partial, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

import loguru
import numpy as np
import openml
import pandas as pd
import yaml
from fedot.api.main import Fedot
from fedot.core.data.data import array_to_input_data
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.metrics_repository import (
    MetricsRepository,
    QualityMetricsEnum,
)
from golem.core.optimisers.fitness import Fitness
from pecapiku import CacheDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing_extensions import Literal

sys.path.insert(0, str(Path(__file__).parents[2]))

from gamlet.approaches.knn_similarity_model_advice import KNNSimilarityModelAdvice
from gamlet.data_preparation.dataset import DatasetIDType, OpenMLDataset, TabularData
from gamlet.data_preparation.datasets_train_test_split import (
    openml_datasets_train_test_split,
)
from gamlet.data_preparation.file_system import get_cache_dir

CONFIGS_DIR = Path(__file__).parent / "configs"

with open(CONFIGS_DIR / "use_configs.yaml", "r") as config_file:
    configs_list = yaml.load(config_file, yaml.Loader)

config = {}
for conf_name in configs_list:
    with open(CONFIGS_DIR / conf_name, "r") as config_file:
        conf = yaml.load(config_file, yaml.Loader)
    intersection = set(config).intersection(set(conf))
    if intersection:
        raise ValueError(f"Parameter values given twice: {conf_name}, {intersection}.")
    config.update(conf)

# Load constants
SEED = config["seed"]
N_DATASETS = config["n_datasets"]
TEST_SIZE = config["test_size"]
TRAIN_TIMEOUT = config["train_timeout"]
TEST_TIMEOUT = config["test_timeout"]
N_BEST_DATASET_MODELS_TO_MEMORIZE = config["n_best_dataset_models_to_memorize"]
ASSESSOR_PARAMS = config["assessor_params"]
ADVISOR_PARAMS = config["advisor_params"]
MF_EXTRACTOR_PARAMS = config["mf_extractor_params"]
COLLECT_METRICS = config["collect_metrics"]
FEDOT_PARAMS = config["fedot_params"]
DATA_TEST_SIZE = config["data_test_size"]
DATA_SPLIT_SEED = config["data_split_seed"]
BASELINE_MODEL = config["baseline_model"]
N_AUTOML_REPETITIONS = config["n_automl_repetitions"]
# Optional values
TMPDIR = config.get("tmpdir")
SAVE_DIR_PREFIX = config.get("save_dir_prefix")

UPDATE_TRAIN_TEST_DATASETS_SPLIT = config.get("update_train_test_datasets_split")

# Postprocess constants
COLLECT_METRICS_ENUM = tuple(map(MetricsRepository.get_metric, COLLECT_METRICS))
COLLECT_METRICS[COLLECT_METRICS.index("neg_log_loss")] = "logloss"


def setup_experiment():
    # Preparation
    experiment_date, experiment_date_iso, experiment_date_for_path = (
        get_current_formatted_date()
    )
    save_dir = get_save_dir(experiment_date_for_path)
    setup_logging(save_dir)
    if TMPDIR:
        os.environ.putenv("TMPDIR", TMPDIR)
    meta_learner_path = save_dir.joinpath("meta_learner.pkl")
    dataset_ids = get_dataset_ids()
    dataset_ids_train, dataset_ids_test = split_datasets(
        dataset_ids, N_DATASETS, UPDATE_TRAIN_TEST_DATASETS_SPLIT
    )
    dataset_ids = dataset_ids_train + dataset_ids_test
    experiment_params_dict = dict(
        experiment_start_date_iso=experiment_date_iso,
        input_config=config,
        dataset_ids=dataset_ids,
        dataset_ids_train=dataset_ids_train,
        dataset_ids_test=dataset_ids_test,
        baseline_pipeline=BASELINE_MODEL,
    )
    save_experiment_params(experiment_params_dict, save_dir)
    return (
        dataset_ids_test,
        dataset_ids_train,
        experiment_date,
        meta_learner_path,
        save_dir,
    )


def setup_logging(save_dir: Path):
    """Creates "log.txt" at the "save_dir" and redirects all logging output to it."""
    loguru.logger.add(save_dir / "file_{time}.log")
    log_file = save_dir.joinpath("log.txt")
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
        level=logging.NOTSET,
    )


def get_current_formatted_date() -> Tuple[datetime, str, str]:
    """Returns current date in the following formats:

    1. datetime
    2. str: ISO
    3. str: ISO compatible with Windows file system path (with "." instead of ":")"""
    time_now = datetime.now()
    time_now_iso = time_now.isoformat(timespec="minutes")
    time_now_for_path = time_now_iso.replace(":", ".")
    return time_now, time_now_iso, time_now_for_path


def get_save_dir(time_now_for_path) -> Path:
    save_dir = (
        get_cache_dir()
        .joinpath("experiments")
        .joinpath("fedot_warm_start")
        .joinpath(f"run_{time_now_for_path}")
    )
    if SAVE_DIR_PREFIX:
        save_dir = save_dir.with_name(SAVE_DIR_PREFIX + save_dir.name)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)

    return save_dir


def get_dataset_ids() -> List[DatasetIDType]:
    dataset_ids = openml.study.get_suite(99).data
    if N_DATASETS is not None:
        dataset_ids = pd.Series(dataset_ids)
        dataset_ids = dataset_ids.sample(n=N_DATASETS, random_state=SEED)
    return list(dataset_ids)


def split_datasets(
    dataset_ids, n_datasets: Optional[int] = None, update_train_test_split: bool = False
) -> Tuple[list, list]:
    split_path = Path(__file__).parent / "train_test_datasets_split.csv"

    if update_train_test_split:
        df_split_datasets = openml_datasets_train_test_split(
            dataset_ids, test_size=TEST_SIZE, seed=SEED
        )
        df_split_datasets.to_csv(split_path)
    else:
        df_split_datasets = pd.read_csv(split_path, index_col=0)

    df_train = df_split_datasets[df_split_datasets["is_train"] == 1]
    df_test = df_split_datasets[df_split_datasets["is_train"] == 0]

    if n_datasets is not None:
        frac = n_datasets / len(df_split_datasets)
        df_train = df_train.sample(frac=frac, random_state=SEED)
        df_test = df_test.sample(frac=frac, random_state=SEED)

    datasets_train = df_train.index.to_list()
    datasets_test = df_test.index.to_list()

    return datasets_train, datasets_test


def evaluate_pipeline(
    pipeline: Pipeline,
    train_data: TabularData,
    test_data: TabularData,
    metrics: Sequence[QualityMetricsEnum] = COLLECT_METRICS_ENUM,
    metric_names: Sequence[str] = COLLECT_METRICS,
    mode: Literal["fitness", "float"] = "float",
) -> Union[Dict[str, float], Tuple[Fitness, Sequence[str]]]:
    """Gets quality metrics for the fitted pipeline.
    The function is based on `Fedot.get_metrics()`

    Returns:
        the values of quality metrics
    """
    train_data = array_to_input_data(train_data.x, train_data.y)
    test_data = array_to_input_data(test_data.x, test_data.y)

    def data_producer():
        yield train_data, test_data

    objective = MetricsObjective(metrics)
    obj_eval = PipelineObjectiveEvaluate(
        objective=objective, data_producer=data_producer, eval_n_jobs=-1
    )

    fitness = obj_eval.evaluate(pipeline)
    if mode == "float":
        metric_values = fitness.values
        metric_values = {
            metric_name: round(value, 3)
            for (metric_name, value) in zip(metric_names, metric_values)
        }
        return metric_values
    if mode == "fitness":
        return fitness, metric_names


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        time_start = timeit.default_timer()
        result = func(*args, **kwargs)
        time_delta = timedelta(seconds=timeit.default_timer() - time_start)
        return result, time_delta

    return wrapper


def fit_evaluate_automl(
    fit_func, evaluate_func
) -> Tuple[Fedot, Dict[str, Any], timedelta]:
    """Runs Fedot evaluation on the dataset, the evaluates the final pipeline on the dataset.."""
    result, fit_time = timed(fit_func)()
    metrics = evaluate_func(result)
    return result, metrics, fit_time


def fit_evaluate_pipeline(
    pipeline, fit_func, evaluate_func
) -> Tuple[Fedot, Dict[str, Any], timedelta]:
    """Runs Fedot evaluation on the dataset, the evaluates the final pipeline on the dataset.."""
    _, fit_time = timed(fit_func)()
    metrics = evaluate_func(pipeline)
    return pipeline, metrics, fit_time


def save_experiment_params(params_dict: Dict[str, Any], save_dir: Path):
    """Save the hyperparameters of the experiment"""
    params_file_path = save_dir.joinpath("parameters.json")
    with open(params_file_path, "w") as params_file:
        json.dump(params_dict, params_file, indent=2)


def save_evaluation(save_dir: Path, dataset, pipeline, **kwargs):
    run_results: Dict[str, Any] = dict(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        model_obj=pipeline,
        model_str=pipeline.descriptive_id,
        task_type="classification",
        **kwargs,
    )
    try:
        histories_dir = save_dir.joinpath("histories")
        models_dir = save_dir.joinpath("models")
        eval_results_path = save_dir.joinpath("evaluation_results.csv")

        histories_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)

        dataset_id = run_results["dataset_id"]
        run_label = run_results["run_label"]
        # define saving paths
        uid = str(uuid4())
        model_path = models_dir.joinpath(f"{dataset_id}_{run_label}_{uid}")
        history_path = histories_dir.joinpath(
            f"{dataset_id}_{run_label}_{uid}_history.json"
        )
        # replace objects with export paths for csv
        run_results["model_path"] = str(model_path)
        run_results.pop("model_obj").save(model_path, create_subdir=False)
        run_results["history_path"] = str(history_path)
        if "history_obj" in run_results:
            history_obj = run_results.pop("history_obj")
            if history_obj is not None:
                history_obj.save(run_results["history_path"])
            run_results["history_obj"] = history_obj

        df_evaluation_properties = pd.DataFrame([run_results])

        if eval_results_path.exists():
            df_results = pd.read_csv(eval_results_path, index_col=None)
            df_results = pd.concat([df_results, df_evaluation_properties])
        else:
            df_results = df_evaluation_properties
        df_results.to_csv(eval_results_path, index=False)

    except Exception as e:
        logging.exception(f'Saving results "{run_results}"')
        if __debug__:
            raise e


def run_fedot_attempt(
    train_data: TabularData,
    test_data: TabularData,
    timeout: float,
    run_label: str,
    repetition: int,
    experiment_date: datetime,
    save_dir: Path,
    initial_assumption: Optional[Sequence[Pipeline]] = None,
    fedot_evaluations_cache=None,
):
    fedot = Fedot(
        timeout=timeout, initial_assumption=initial_assumption, **FEDOT_PARAMS
    )
    fit_func = partial(fedot.fit, features=train_data.x, target=train_data.y)
    evaluate_func = partial(
        evaluate_pipeline, train_data=train_data, test_data=test_data
    )
    run_date = datetime.now()
    cache_key = f"{run_label}_{train_data.id}_{timeout}_{repetition}"
    with fedot_evaluations_cache as cache_dict:
        cached_run = cache_dict[cache_key]
        if cached_run:
            fedot = cached_run["fedot"]
            pipeline = cached_run["pipeline"]
            metrics = cached_run["metrics"]
            fit_time = cached_run["fit_time"]
        else:
            #         pipeline, metrics, fit_time = fit_evaluate_automl(fit_func=fit_func, evaluate_func=evaluate_func)
            #         cached_run = dict(
            #             fedot=fedot,
            #             pipeline=pipeline,
            #             metrics=metrics,
            #             fit_time=fit_time,
            #         )
            #         cache_dict[cache_key] = cached_run
            pipeline, metrics, fit_time = fit_evaluate_automl(
                fit_func=fit_func, evaluate_func=evaluate_func
            )
    eval_result = dict(
        dataset=train_data.dataset,
        run_label=run_label,
        pipeline=pipeline,
        automl_time_min=fit_time.total_seconds() / 60,
        automl_timeout_min=fedot.params.timeout,
        generations_count=fedot.history.generations_count,
        history_obj=fedot.history,
        run_data=run_date,
        experiment_date=experiment_date,
        save_dir=save_dir,
        **metrics,
    )
    return eval_result


def run_pipeline(
    train_data: TabularData,
    test_data: TabularData,
    pipeline: Pipeline,
    run_label: str,
    experiment_date: datetime,
    save_dir: Path,
):
    train_data_for_fedot = array_to_input_data(train_data.x, train_data.y)
    fit_func = partial(pipeline.fit, train_data_for_fedot)
    evaluate_func = partial(
        evaluate_pipeline, train_data=train_data, test_data=test_data
    )
    run_date = datetime.now()
    pipeline, metrics, fit_time = fit_evaluate_pipeline(
        pipeline=pipeline, fit_func=fit_func, evaluate_func=evaluate_func
    )
    save_evaluation(
        dataset=train_data.dataset,
        run_label=run_label,
        pipeline=pipeline,
        automl_time_min=0,
        pipeline_fit_time_sec=fit_time.total_seconds(),
        automl_timeout_min=0,
        meta_learning_time_sec=0,
        run_data=run_date,
        experiment_date=experiment_date,
        save_dir=save_dir,
        **metrics,
    )
    return pipeline


def get_datasets_eval_funcs(dataset_ids_train, dataset_splits):
    dataset_eval_funcs = []
    for dataset_id in dataset_ids_train:
        split = dataset_splits[dataset_id]
        train_data, test_data = split["train"], split["test"]
        model_eval_func = partial(
            evaluate_pipeline,
            train_data=train_data,
            test_data=test_data,
            mode="fitness",
        )
        dataset_eval_funcs.append(model_eval_func)
    return dataset_eval_funcs


def get_datasets_data_splits(dataset_ids):
    dataset_splits = {}
    for dataset_id in dataset_ids:
        dataset = OpenMLDataset(dataset_id)
        dataset_data = dataset.get_data()
        if isinstance(dataset_data.y[0], bool):
            dataset_data.y = np.array(list(map(str, dataset_data.y)))
        idx_train, idx_test = train_test_split(
            range(len(dataset_data.y)),
            test_size=DATA_TEST_SIZE,
            stratify=dataset_data.y,
            shuffle=True,
            random_state=DATA_SPLIT_SEED,
        )
        train_data, test_data = dataset_data[idx_train], dataset_data[idx_test]
        dataset_splits[dataset_id] = dict(train=train_data, test=test_data)
    return dataset_splits


def evaluate_fedot_on_dataset(
    train_data: TabularData,
    test_data: TabularData,
    timeout: float,
    run_label: str,
    experiment_date: datetime,
    save_dir: Path,
    fedot_evaluations_cache: CacheDict,
    initial_assumption: Optional[Sequence[Pipeline]] = None,
    meta_learning_time: Optional[timedelta] = None,
):
    meta_learning_time = meta_learning_time or timedelta(0)
    dataset = train_data.dataset

    eval_results = []
    for repetition in range(N_AUTOML_REPETITIONS):
        try:
            eval_result, time_delta = timed(run_fedot_attempt)(
                train_data,
                test_data,
                timeout,
                run_label,
                repetition,
                experiment_date,
                save_dir,
                initial_assumption,
                fedot_evaluations_cache,
            )
            time_limit = timedelta(minutes=timeout * 2)
            if time_delta > time_limit:
                logging.warning(
                    f'Dataset "{dataset.id}" TIMEOUT REACHED, {time_delta}.'
                )
                return None

            eval_results.append(eval_result)
        except Exception as e:
            logging.warning(f'Dataset "{dataset.id}" skipepd: {e}')
            logging.exception(f'Dataset "{dataset.id}"')
            if __debug__:
                raise e
            return None

    generations_total = sum(
        map(lambda ev_res: ev_res["history_obj"].generations_count, eval_results)
    )
    if generations_total == 0:
        logging.warning(f'Dataset "{dataset.id}": zero generations obtained.')
        return None

    for eval_result in eval_results:
        eval_result["meta_learning_time_sec"] = meta_learning_time.total_seconds()
        save_evaluation(**eval_result)

        histories = list(map(lambda r: r["history_obj"], eval_results))

        return histories


@loguru.logger.catch
def main():
    (
        dataset_ids_test,
        dataset_ids_train,
        experiment_date,
        meta_learner_path,
        save_dir,
    ) = setup_experiment()

    dataset_splits = get_datasets_data_splits(dataset_ids_test + dataset_ids_train)

    algorithm = KNNSimilarityModelAdvice(
        N_BEST_DATASET_MODELS_TO_MEMORIZE,
        MF_EXTRACTOR_PARAMS,
        ASSESSOR_PARAMS,
        ADVISOR_PARAMS,
    )
    # Experiment start
    # knowledge_base = {dataset_id: [] for dataset_id in dataset_ids_train}
    knowledge_base = {}
    skipped_datasets = set()
    fedot_evaluations_cache = CacheDict(get_cache_dir() / "fedot_runs.pkl")
    # fedot_evaluations_cache = None
    # evaluate_fedot_on_dataset_cached = CacheDict.decorate(evaluate_fedot_on_dataset, get_cache_dir() / 'fedot_runs.pkl', inner_key='train_data.id')
    description = "FEDOT, all datasets ({dataset_id})"
    for dataset_id in (pbar := tqdm(dataset_ids_train + dataset_ids_test, description)):
        pbar.set_description(description.format(dataset_id=dataset_id))
        train_data, test_data = (
            dataset_splits[dataset_id]["train"],
            dataset_splits[dataset_id]["test"],
        )
        run_label = "FEDOT"
        timeout = TRAIN_TIMEOUT if dataset_id in dataset_ids_test else TEST_TIMEOUT
        histories = evaluate_fedot_on_dataset(
            train_data,
            test_data,
            timeout,
            run_label,
            experiment_date,
            save_dir,
            fedot_evaluations_cache,
        )
        if histories is not None:
            if dataset_id in dataset_ids_train:
                knowledge_base[dataset_id] = histories
            continue
        # Error processing - throw the dataset out
        skipped_datasets.add(dataset_id)
        if dataset_id in dataset_ids_train:
            del dataset_ids_train[dataset_ids_train.index(dataset_id)]
        else:
            del dataset_ids_test[dataset_ids_test.index(dataset_id)]

    with open(save_dir / "skipped_datasets.txt", "w") as f:
        f.write("\n".join(map(str, skipped_datasets)))

    ###############################
    kb_datasets_data = [
        OpenMLDataset(dataset).get_data() for dataset in knowledge_base.keys()
    ]
    # datasets_eval_funcs = get_datasets_eval_funcs(dataset_ids_train, dataset_splits)
    datasets_eval_funcs = None
    kb_histories = list(knowledge_base.values())
    ###############################

    # Meta-Learning
    algorithm.fit(kb_datasets_data, kb_histories, datasets_eval_funcs)
    for dataset_id in dataset_ids_train:
        if dataset_id not in algorithm.data.dataset_ids:
            skipped_datasets.add(dataset_id)
            del dataset_ids_train[dataset_ids_train.index(dataset_id)]
    with open(save_dir / "skipped_datasets.txt", "w") as f:
        f.write("\n".join(map(str, skipped_datasets)))

    with open(meta_learner_path, "wb") as meta_learner_file:
        pickle.dump(algorithm, meta_learner_file)
    # Application
    # evaluate_metafedot_on_dataset_cached = CacheDict.decorate(evaluate_fedot_on_dataset, get_cache_dir() / 'metafedot_runs.pkl', inner_key='train_data.id')
    fedot_evaluations_cache = CacheDict(get_cache_dir() / "metafedot_runs.pkl")
    description = "FEDOT, test datasets ({dataset_id})"
    for dataset_id in (pbar := tqdm(dataset_ids_test, description)):
        pbar.set_description(description.format(dataset_id=dataset_id))
        train_data, test_data = (
            dataset_splits[dataset_id]["train"],
            dataset_splits[dataset_id]["test"],
        )
        # Run meta AutoML
        # 1
        try:
            initial_assumptions, meta_learning_time = timed(algorithm.predict)(
                [train_data]
            )
            if not initial_assumptions:
                raise ValueError("No intial assumptions.")
        except Exception:
            logging.exception(
                f'Dataset "{dataset_id}" skipepd, meta learner could not predict: {e}'
            )
            skipped_datasets.add(dataset_id)
            del dataset_ids_test[dataset_ids_test.index(dataset_id)]
            continue

        initial_assumptions = initial_assumptions[0]
        assumption_pipelines = [model.predictor for model in initial_assumptions]
        # 2
        baseline_pipeline = PipelineBuilder().add_node(BASELINE_MODEL).build()
        run_label = "MetaFEDOT"
        try:
            histories = evaluate_fedot_on_dataset(
                train_data,
                test_data,
                TEST_TIMEOUT,
                run_label,
                experiment_date,
                save_dir,
                fedot_evaluations_cache,
                assumption_pipelines,
                meta_learning_time,
            )
            if histories is None:
                raise ValueError("No results.")
        except Exception as e:
            logging.exception(
                f'Dataset "{dataset_id}" skipepd, meta fedot could not finish: {e}'
            )
            skipped_datasets.add(dataset_id)
            del dataset_ids_test[dataset_ids_test.index(dataset_id)]
            continue
        # Fit & evaluate simple baseline
        run_label = "simple baseline"
        try:
            run_pipeline(
                train_data,
                test_data,
                baseline_pipeline,
                run_label,
                experiment_date,
                save_dir,
            )
        except Exception as e:
            logging.exception(f'Test dataset "{dataset_id}", {run_label}')
            if __debug__:
                raise e
        # Fit & evaluate initial assumptions
        for i, assumption in enumerate(initial_assumptions):
            try:
                pipeline = assumption.predictor
                run_label = f"MetaFEDOT - initial assumption {i}"
                run_pipeline(
                    train_data,
                    test_data,
                    pipeline,
                    run_label,
                    experiment_date,
                    save_dir,
                )
            except Exception as e:
                logging.exception(f'Test dataset "{dataset_id}", {run_label}')
                if __debug__:
                    raise e

    with open(save_dir / "skipped_datasets.txt", "w") as f:
        f.write("\n".join(map(str, skipped_datasets)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Exception at main().")
        raise e
