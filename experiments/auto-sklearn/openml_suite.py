import csv
import os
import pickle
import re
import time

import numpy as np
import json

import autosklearn.classification
import autosklearn.ensembles
from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import Balancing
from autosklearn.pipeline.components.data_preprocessing import DataPreprocessorChoice
from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.components.classification import AutoSklearnClassificationAlgorithm, ClassifierChoice

# from experiments.fedot_warm_start.run import fetch_openml_data, mock_data_fetching
from sklearn import model_selection, metrics
from sklearn import ensemble
from sklearn.base import ClassifierMixin


# class AutoSklearnEncoder(json.JSONEncoder):
#     def default(self, o):
#         # if isinstance(o, dict):
#         #     return json.dumps(o)
#         if isinstance(o, ClassifierChoice):
#             return repr(o.choice.estimator)
#         # if isinstance(o, (DataPreprocessorChoice, FeaturePreprocessorChoice)):
#         #     return None
#         elif isinstance(o, ClassifierMixin):
#             return re.sub(r'\s{2,}', ' ', repr(o))
#         # elif isinstance(o, Balancing):
#         #     return repr(o)
#         elif isinstance(o, np.integer):
#             return int(o)
#         elif isinstance(o, np.floating):
#             return float(o)


class AutoSklearnBaseline:

    def __init__(self):
        pass

    @staticmethod
    def main():
        openml_data = None
        # dataset_names = [dataset.name for dataset in openml_data]

        # train_data_names, test_data_names = model_selection.train_test_split(
        #     [dataset.name for dataset in openml_data],
        #     test_size=0.2,
        #     random_state=42
        # )
        # train_ds_names, test_ds_names = ds_names

        # ds_ids, datasets = ds_with_ids

        # for ds_name in train_ds_names:

        for iteration, dataset in enumerate(openml_data):
            print(f"Fetched data name: {dataset.name}")
            dataset = dataset.from_cache()

            # estimator = autosklearn.classification.AutoSklearnClassifier(
            #     ensemble_class=autosklearn.ensembles.SingleBest,
            #     time_left_for_this_task=600
            # )
            estimator = ensemble.HistGradientBoostingClassifier()

            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                dataset.x,
                dataset.y,
                test_size=0.2,
                random_state=42
            )

            fitting_start_time = time.time()
            pipeline = estimator.fit(X_train, y_train)
            fitting_end_time = time.time() - fitting_start_time
            # print(f"Fitting time is {fitting_end_time}sec")

            inference_start_time = time.time()
            predictions = estimator.predict(X_test)
            inference_end_time = time.time() - inference_start_time

            prediction_probabilities = estimator.predict_proba(X_test)

            is_multi_classification_problem = True if len(set(predictions)) > 2 else False
            # print(f"Inference time is {inference_end_time}sec")
            # roc_auc_score = metrics.roc_auc_score(y_test, predictions)

            # autosklearn_ensemble = pipeline.show_models()
            # formatted_ensemble = {
            #     model_id: {
            #         'rank': autosklearn_ensemble[model_id].get('rank'),
            #         'cost': float(f"{autosklearn_ensemble[model_id].get('cost'):.3f}"),
            #         'ensemble_weight': autosklearn_ensemble[model_id].get('ensemble_weight'),
            #         'model': autosklearn_ensemble[model_id].get('sklearn_classifier')
            #     } for model_id in autosklearn_ensemble.keys()
            # }

            # best_single_model = list(pipeline.show_models().values())[0].get('sklearn_classifier')
            best_single_model = repr(pipeline)
            # encoded_ensemble = str(formatted_ensemble).encode('base64')

            # print(f"y_test is {predictions}")

            general_run_info = {
                # 'id': iteration + 1,
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'run_label': 'Hist gradient boosting classifier'
            }
            average = 'macro' if is_multi_classification_problem else 'binary'
            model_dependent_run_info = {
                'roc_auc': -1 * float(f"{metrics.roc_auc_score(y_test, prediction_probabilities if is_multi_classification_problem else predictions, multi_class='ovr'):.3f}"),
                'f1': -1 * float(f"{metrics.f1_score(y_test, predictions, average=average):.3f}"),
                'accuracy': -1 * float(f"{metrics.accuracy_score(y_test, predictions):.3f}"),
                'logloss': float(f"{metrics.log_loss(y_test, prediction_probabilities if is_multi_classification_problem else predictions):.3f}"),
                'precision': -1 * float(f"{metrics.precision_score(y_test, predictions, average=average):.3f}"),
                'fit_time': float(f'{fitting_end_time:.1f}'),
                'inference_time': float(f'{inference_end_time:.1f}'),
                # 'model_str': re.sub(r'\s{2,}', ' ', repr(best_single_model))
                'model_str': None
            }
            results = {**general_run_info, **model_dependent_run_info}

            # for key in autosklearn_ensemble.keys():
            #     ensemble_model = autosklearn_ensemble[key]
            #     formatted_ensemble = results['ensemble']
            #     for model_id in formatted_ensemble.keys():
            #         formatted_ensemble[model_id] = ensemble_model.get("rank", None)

            # pickle.dump(pipeline.show_models(), open("results.pickle", "wb"))

            # print(type(pipeline.show_models().get(list(pipeline.show_models().keys())[0]).get("classifier")))

            # knowledge_base_path = os.path.dirname('knowledge_base_0')

            with open('experimental_data.csv', 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                # if iteration == 0:
                #     writer.writerow(results.keys())
                writer.writerow(results.values())

if __name__ == '__main__':
    AutoSklearnBaseline.main()

