import csv
import time

from typing import Any, Tuple, Dict

import numpy as np
import logging

import autosklearn.classification
import autosklearn.ensembles

from sklearn import model_selection, metrics

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.models_loaders import KnowledgeBaseModelsLoader
from autosklearn.classification import AutoSklearnClassifier



class AutoSklearnBaseline:
    def __init__(self, estimator_ensemble_type, time_limit):
        self.estimator = AutoSklearnClassifier(
            ensemble_class=estimator_ensemble_type,
            time_left_for_this_task=time_limit,
        )
        self.knowledge_base_loader = KnowledgeBaseModelsLoader()

    def make_quality_metric_estimates(self, y, predictions, prediction_proba, is_multi_label):
        """ Compute roc_auc, f1, accuracy, log_loss and precision scores. """
        results = {
            'roc_auc': -1 * float(
                "{:.3f}".format(
                    metrics.roc_auc_score(
                        y,
                        prediction_proba if is_multi_label else predictions,
                        multi_class='ovr'
                    )
                )
            ),
            'f1': -1 * float(
                "{:.3f}".format(
                    metrics.f1_score(
                        y,
                        predictions,
                        average='macro' if is_multi_label else 'binary'
                    )
                )
            ),
            'accuracy': -1 * float(
                "{:.3f}".format(
                    metrics.accuracy_score(
                        y,
                        predictions
                    )
                )
            ),
            'logloss': float(
                "{:.3f}".format(
                    metrics.log_loss(
                        y,
                        prediction_proba if is_multi_label else predictions
                    )
                )
            ),
            'precision': -1 * float(
                "{:.3f}".format(
                    metrics.precision_score(
                        y,
                        predictions,
                        average='macro' if is_multi_label else 'binary',
                        labels=np.unique(predictions)
                    )
                )
            )
        }
        return results

    def run(self):
        """ Fit auto-sklearn meta-optimizer to knowledge base datasets and output a single best model. """
        dataset_ids_to_load = [
            dataset_id for dataset_id in self.knowledge_base_loader
                                             .parse_datasets('test')
                                             .loc[:, 'dataset_id']
        ]
        dataset_ids_to_load = [dataset_ids_to_load[dataset_ids_to_load.index(41166)]]

        loaded_datasets = OpenMLDatasetsLoader().load(dataset_ids_to_load)

        for iteration, dataset in enumerate(loaded_datasets):
            logging.log(logging.INFO, f"Loaded dataset name: {dataset.name}")
            dataset = dataset.from_cache()

            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                dataset.x,
                dataset.y,
                test_size=0.2,
                random_state=42,
                stratify=dataset.y
            )

            fitting_start_time = time.time()
            ensemble = self.estimator.fit(X_train, y_train)
            fitting_time = time.time() - fitting_start_time
            logging.log(logging.INFO, f"Fitting time is {fitting_time}sec")

            inference_start_time = time.time()
            predicted_results = self.estimator.predict(X_test)
            inference_time = time.time() - inference_start_time
            logging.log(logging.INFO, f"Inference time is {inference_time}sec")

            predicted_probabilities = self.estimator.predict_proba(X_test)

            best_single_model = list(ensemble.show_models().values())[0].get('sklearn_classifier')

            # autosklearn_ensemble = pipeline.show_models()
            # formatted_ensemble = {
            #     model_id: {
            #         'rank': autosklearn_ensemble[model_id].get('rank'),
            #         'cost': float(f"{autosklearn_ensemble[model_id].get('cost'):.3f}"),
            #         'ensemble_weight': autosklearn_ensemble[model_id].get('ensemble_weight'),
            #         'model': autosklearn_ensemble[model_id].get('sklearn_classifier')
            #     } for model_id in autosklearn_ensemble.keys()
            # }

            general_run_info = {
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'run_label': 'Auto-sklearn',
            }

            is_multilabel_classification = True if len(set(predicted_results)) > 2 else False
            quality_metric_estimates = self.make_quality_metric_estimates(
                y_test,
                predicted_results,
                predicted_probabilities,
                is_multilabel_classification
            )

            model_dependent_run_info = {
                'fit_time': float(f'{fitting_time:.1f}'),
                'inference_time': float(f'{inference_time:.1f}'),
                'model_str': repr(best_single_model)
            }

            results = {**general_run_info, **quality_metric_estimates, **model_dependent_run_info}

            # for key in autosklearn_ensemble.keys():
            #     ensemble_model = autosklearn_ensemble[key]
            #     formatted_ensemble = results['ensemble']
            #     for model_id in formatted_ensemble.keys():
            #         formatted_ensemble[model_id] = ensemble_model.get("rank", None)

            with open('experimental_data.csv', 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(results.values())


if __name__ == '__main__':
    AutoSklearnBaseline(autosklearn.ensembles.SingleBest, 600).run()
