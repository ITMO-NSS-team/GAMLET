import functools
import json
import logging
import timeit
from pathlib import Path

import yaml

from datetime import datetime
from itertools import chain
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np
import openml
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.quality_metrics_repository import QualityMetricsEnum, MetricsRepository
from fedot.core.validation.split import tabular_cv_generator
from golem.core.log import Log
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


from meta_automl.data_preparation.dataset import OpenMLDataset, DatasetData, DatasetBase
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.datasets_train_test_split import openml_datasets_train_test_split
from meta_automl.data_preparation.file_system import get_cache_dir
from meta_automl.data_preparation.meta_features_extractors import PymfeExtractor
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import KNeighborsBasedSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import DiverseFEDOTPipelineAdvisor


CONFIG_PATH = Path(__file__).parent.joinpath('config.yaml')


with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.load(config_file, yaml.Loader)

# Load constants
SEED = config['seed']
N_DATASETS = config['n_datasets']
TEST_SIZE = config['test_size']
TRAIN_TIMEOUT = config['train_timeout']
TEST_TIMEOUT = config['test_timeout']
N_BEST_DATASET_MODELS_TO_MEMORIZE = config['n_best_dataset_models_to_memorize']
N_CLOSEST_DATASETS_TO_PROPOSE = config['n_closest_datasets_to_propose']
MINIMAL_DISTANCE_BETWEEN_ADVISED_MODELS = config['minimal_distance_between_advised_models']
N_BEST_MODELS_TO_ADVISE = config['n_best_models_to_advise']
MF_EXTRACTOR_PARAMS = config['mf_extractor_params']
COLLECT_METRICS = config['collect_metrics']
COMMON_FEDOT_PARAMS = config['common_fedot_params']
BASELINE_MODEL = config['baseline_model']

# Postprocess constants
COLLECT_METRICS_ENUM = tuple(map(MetricsRepository.metric_by_id, COLLECT_METRICS))
COLLECT_METRICS[COLLECT_METRICS.index('neg_log_loss')] = 'logloss'
COMMON_FEDOT_PARAMS['seed'] = SEED


def setup_logging(save_dir: Path):
    """ Creates "log.txt" at the "save_dir" and redirects all logging output to it. """
    log_file = save_dir.joinpath('log.txt')
    Log(log_file=log_file)
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        force=True,
    )


def get_current_formatted_date() -> (datetime, str, str):
    """ Returns current date in the following formats:

        1. datetime
        2. str: ISO
        3. str: ISO compatible with Windows file system path (with "." instead of ":") """
    time_now = datetime.now()
    time_now_iso = time_now.isoformat(timespec="minutes")
    time_now_for_path = time_now_iso.replace(":", ".")
    return time_now, time_now_iso, time_now_for_path


def get_save_dir(time_now_for_path) -> Path:
    save_dir = get_cache_dir(). \
        joinpath('experiments').joinpath('fedot_warm_start').joinpath(f'run_{time_now_for_path}')
    save_dir.mkdir(parents=True)
    return save_dir


def fetch_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, OpenMLDataset]]:
    """Returns dictionary with dataset names and cached datasets downloaded from OpenML."""

    dataset_ids = openml.study.get_suite(99).data
    if N_DATASETS is not None:
        dataset_ids = pd.Series(dataset_ids)
        dataset_ids = dataset_ids.sample(n=N_DATASETS, random_state=SEED)

    df_split_datasets = openml_datasets_train_test_split(dataset_ids, test_size=TEST_SIZE, seed=SEED)
    df_datasets_train = df_split_datasets[df_split_datasets['is_train'] == 1]
    df_datasets_test = df_split_datasets[df_split_datasets['is_train'] == 0]

    datasets = {dataset.id_: dataset for dataset in OpenMLDatasetsLoader().load(dataset_ids)}
    return df_datasets_train, df_datasets_test, datasets


def evaluate_pipeline(pipeline: Pipeline,
                      input_data: InputData,
                      metrics: Sequence[QualityMetricsEnum] = COLLECT_METRICS_ENUM,
                      metric_names: Sequence[str] = COLLECT_METRICS) -> Dict[str, float]:
    """Gets quality metrics for the fitted pipeline.
    The function is based on `Fedot.get_metrics()`

    Returns:
        the values of quality metrics
    """
    data_producer = functools.partial(tabular_cv_generator, input_data, 10, StratifiedKFold)

    objective = MetricsObjective(metrics)
    obj_eval = PipelineObjectiveEvaluate(objective=objective,
                                         data_producer=data_producer,
                                         eval_n_jobs=-1)

    metric_values = obj_eval.evaluate(pipeline).values
    metric_values = {metric_name: round(value, 3) for (metric_name, value) in zip(metric_names, metric_values)}

    return metric_values


def fit_offline_meta_learning_components(best_models_per_dataset_id: Dict[int, Sequence[Model]]) \
        -> (KNeighborsBasedSimilarityAssessor, PymfeExtractor, DiverseFEDOTPipelineAdvisor):
    dataset_ids = list(best_models_per_dataset_id.keys())
    # Meta Features
    extractor = PymfeExtractor(extractor_params=MF_EXTRACTOR_PARAMS)
    meta_features_train = extractor.extract(dataset_ids, fill_input_nans=True)
    meta_features_train = meta_features_train.fillna(0)
    # Datasets similarity
    data_similarity_assessor = KNeighborsBasedSimilarityAssessor(
        n_neighbors=min(len(dataset_ids), N_CLOSEST_DATASETS_TO_PROPOSE))
    data_similarity_assessor.fit(meta_features_train, dataset_ids)
    # Model advisor
    model_advisor = DiverseFEDOTPipelineAdvisor(data_similarity_assessor, n_best_to_advise=N_BEST_MODELS_TO_ADVISE,
                                                minimal_distance=MINIMAL_DISTANCE_BETWEEN_ADVISED_MODELS)
    model_advisor.fit(best_models_per_dataset_id)
    return extractor, model_advisor


def transform_data_for_fedot(data: DatasetData) -> (np.array, np.array):
    x = data.x
    y = data.y
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return x, y


def fit_fedot(dataset: OpenMLDataset, timeout: float, run_label: str, initial_assumption=None) \
        -> (Fedot, Dict[str, Any]):
    """ Runs Fedot evaluation on the dataset, the evaluates the final pipeline on the dataset.
     Returns Fedot instance & properties of the run along with the evaluated metrics. """
    x, y = transform_data_for_fedot(dataset.get_data(dataset_format='array'))

    time_start = timeit.default_timer()
    fedot = Fedot(timeout=timeout, initial_assumption=initial_assumption, **COMMON_FEDOT_PARAMS)
    fedot.fit(x, y)
    automl_time = timeit.default_timer() - time_start

    metrics = evaluate_pipeline(fedot.current_pipeline, fedot.train_data)
    pipeline = fedot.current_pipeline
    run_results = get_result_data_row(dataset=dataset, run_label=run_label, pipeline=pipeline,
                                      automl_time_sec=automl_time, automl_timeout_min=fedot.params.timeout,
                                      history_obj=fedot.history, **metrics)
    return fedot, run_results


def get_result_data_row(dataset: OpenMLDataset, run_label: str, pipeline, history_obj=None, automl_time_sec=0.,
                        automl_timeout_min=0., **metrics) -> Dict[str, Any]:
    run_results = dict(dataset_id=dataset.id_,
                       dataset_name=dataset.name,
                       run_label=run_label,
                       model_obj=pipeline,
                       model_str=pipeline.descriptive_id,
                       history_obj=history_obj,
                       automl_time_sec=automl_time_sec,
                       automl_timeout_min=automl_timeout_min,
                       task_type='classification',
                       **metrics)
    return run_results


def extract_best_models_from_history(dataset: DatasetBase, history: OptHistory) -> List[Model]:
    if history.individuals:
        best_individuals = sorted(chain(*history.individuals),
                                  key=lambda ind: ind.fitness,
                                  reverse=True)
        best_individuals = list({ind.graph.descriptive_id: ind for ind in best_individuals}.values())
        best_models = []
        for individual in best_individuals[:N_BEST_DATASET_MODELS_TO_MEMORIZE]:
            pipeline = PipelineAdapter().restore(individual.graph)
            model = Model(pipeline, individual.fitness, history.objective.metric_names[0], dataset)
            best_models.append(model)
    else:
        pipeline = PipelineAdapter().restore(history.tuning_result)
        best_models = [Model(pipeline, SingleObjFitness(), history.objective.metric_names[0], dataset)]

    return best_models


def save_experiment_params(params_dict: Dict[str, Any], save_dir: Path):
    """ Save the hyperparameters of the experiment """
    params_file_path = save_dir.joinpath('parameters.json')
    with open(params_file_path, 'w') as params_file:
        json.dump(params_dict, params_file, indent=2)


def save_evaluation(evaluation_properties: Dict[str, Any], run_date: datetime, experiment_date: datetime,
                    save_dir: Path):
    histories_dir = save_dir.joinpath('histories')
    models_dir = save_dir.joinpath('models')
    eval_results_path = save_dir.joinpath('evaluation_results.csv')

    histories_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    try:
        evaluation_properties['experiment_date'] = experiment_date
        evaluation_properties['run_date'] = run_date
        dataset_id = evaluation_properties['dataset_id']
        run_label = evaluation_properties['run_label']
        # define saving paths
        model_path = models_dir.joinpath(f'{dataset_id}_{run_label}')
        history_path = histories_dir.joinpath(f'{dataset_id}_{run_label}_history.json')
        # replace objects with export paths for csv
        evaluation_properties['model_path'] = str(model_path)
        evaluation_properties.pop('model_obj').save(model_path)
        evaluation_properties['history_path'] = str(history_path)
        history_obj = evaluation_properties.pop('history_obj')
        if history_obj is not None:
            history_obj.save(evaluation_properties['history_path'])

        df_evaluation_properties = pd.DataFrame([evaluation_properties])

        if eval_results_path.exists():
            df_results = pd.read_csv(eval_results_path)
            df_results = pd.concat([df_results, df_evaluation_properties])
        else:
            df_results = df_evaluation_properties
        df_results.to_csv(eval_results_path, index=False)

    except Exception:
        logging.exception(f'Saving results "{evaluation_properties}"')


def main():
    experiment_date, experiment_date_iso, experiment_date_for_path = get_current_formatted_date()
    save_dir = get_save_dir(experiment_date_for_path)
    setup_logging(save_dir)
    progress_file_path = save_dir.joinpath('progress.txt')

    df_datasets_train, df_datasets_test, datasets_dict = fetch_datasets()

    dataset_ids = list(datasets_dict.keys())
    dataset_ids_train = df_datasets_train.index.to_list()
    dataset_ids_test = df_datasets_test.index.to_list()

    dataset_names_train = df_datasets_train['dataset_name'].to_list()
    dataset_names_test = df_datasets_test['dataset_name'].to_list()

    datasets_dict_test = dict(filter(lambda item: item[0] in dataset_ids_test, datasets_dict.items()))

    experiment_params_dict = dict(
            experiment_start_date_iso=experiment_date_iso,
            input_config=config,
            dataset_ids=dataset_ids,
            dataset_ids_train=dataset_ids_train,
            dataset_names_train=dataset_names_train,
            dataset_ids_test=dataset_ids_test,
            dataset_names_test=dataset_names_test,
            baseline_pipeline=BASELINE_MODEL,
        )
    save_experiment_params(experiment_params_dict, save_dir)

    best_models_per_dataset = {}
    with open(progress_file_path, 'a') as progress_file:
        for dataset_id, dataset in tqdm(datasets_dict.items(), 'FEDOT, all datasets', file=progress_file):
            try:
                timeout = TRAIN_TIMEOUT if dataset_id in dataset_ids_train else TEST_TIMEOUT
                run_date = datetime.now()
                fedot, run_results = fit_fedot(dataset=dataset, timeout=timeout, run_label='FEDOT')
                save_evaluation(run_results, run_date, experiment_date, save_dir)
                # TODO:
                #   x Turn the tuned pipeline into a model (evaluate its fitness on the data)
                #   x Evaluate historical pipelines on the data instead of using fitness
                #   x Start FEDOT `N_BEST_DATASET_MODELS_TO_MEMORIZE` times, but not in one run

                # Filter out unique individuals with the best fitness
                history = fedot.history
                best_models = extract_best_models_from_history(dataset, history)
                best_models_per_dataset[dataset_id] = best_models
            except Exception:
                logging.exception(f'Train dataset "{dataset_id}"')

    mf_extractor, model_advisor = fit_offline_meta_learning_components(best_models_per_dataset)

    with open(progress_file_path, 'a') as progress_file:
        for dataset_id, dataset in tqdm(datasets_dict_test.items(), 'MetaFEDOT, Test datasets', file=progress_file):
            try:
                # Run meta AutoML
                # 1
                time_start = timeit.default_timer()
                meta_features = mf_extractor.extract([dataset],
                                                     fill_input_nans=True, use_cached=False, update_cached=True)
                meta_features = meta_features.fillna(0)
                meta_learning_time_sec = timeit.default_timer() - time_start
                initial_assumptions = model_advisor.predict(meta_features)[0]
                assumption_pipelines = [model.predictor for model in initial_assumptions]
                # 2
                run_date = datetime.now()
                fedot_meta, fedot_meta_results = fit_fedot(dataset=dataset, timeout=TEST_TIMEOUT, run_label='MetaFEDOT',
                                                           initial_assumption=assumption_pipelines)
                fedot_meta_results['meta_learning_time_sec'] = meta_learning_time_sec
                save_evaluation(fedot_meta_results, run_date, experiment_date, save_dir)

                # Fit & evaluate simple baseline
                baseline_pipeline = PipelineBuilder().add_node(BASELINE_MODEL).build()
                run_date = datetime.now()
                baseline_metrics = evaluate_pipeline(baseline_pipeline, fedot_meta.train_data)
                baseline_res = get_result_data_row(dataset=dataset, run_label=f'simple baseline {BASELINE_MODEL}',
                                                   pipeline=baseline_pipeline,
                                                   **baseline_metrics)
                save_evaluation(baseline_res, run_date, experiment_date, save_dir)

                # Fit & evaluate initial assumptions
                for i, assumption in enumerate(initial_assumptions):
                    pipeline = assumption.predictor
                    run_date = datetime.now()
                    assumption_metrics = evaluate_pipeline(pipeline, fedot_meta.train_data)
                    assumption_res = get_result_data_row(dataset=dataset,
                                                         run_label=f'MetaFEDOT - initial assumption {i}',
                                                         pipeline=pipeline, **assumption_metrics)
                    save_evaluation(assumption_res, run_date, experiment_date, save_dir)
            except Exception:
                logging.exception(f'Test dataset "{dataset_id}"')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception('Main level caught an error.')
        raise
