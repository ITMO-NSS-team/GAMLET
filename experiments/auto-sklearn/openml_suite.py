import pickle
import re

import numpy as np
import json

import autosklearn.classification
from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import Balancing
from autosklearn.pipeline.components.data_preprocessing import DataPreprocessorChoice
from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.components.classification import AutoSklearnClassificationAlgorithm, ClassifierChoice

from experiments.fedot_warm_start.run import prepare_data
from sklearn import model_selection, metrics
from sklearn.base import ClassifierMixin


class AutoSklearnEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ClassifierChoice):
            return repr(o.choice.estimator)
        # if isinstance(o, (DataPreprocessorChoice, FeaturePreprocessorChoice)):
        #     return None
        if isinstance(o, ClassifierMixin):
            return re.sub(r'\s{2,}', ' ', repr(o))
        elif isinstance(o, Balancing):
            return repr(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)


class AutoSklearnValidator:

    def __init__(self):
        pass

    @staticmethod
    def main():
        ds_with_ids, ds_names = prepare_data()
        train_ds_names, test_ds_names = ds_names

        ds_ids, datasets = ds_with_ids

        for ds_name in train_ds_names:
        # if train_ds_names[0] is not None:
            print("Sanity check")
            dataset = datasets[ds_name].from_cache()

            # cannot wait longer because of the slow data fetching, issue#9
            estimator = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=60
            )

            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                dataset.x,
                dataset.y,
                test_size=0.2,
                random_state=42
            )

            pipeline = estimator.fit(X_train, y_train)

            predictions = estimator.predict(X_test)

            quality_estimation = metrics.roc_auc_score(y_test, predictions)

            results = {
                'ensemble': pipeline.show_models(),
                'score': quality_estimation
            }

            # pickle.dump(pipeline.show_models(), open("results.pickle", "wb"))

            # print(type(pipeline.show_models().get(list(pipeline.show_models().keys())[0]).get("classifier")))

            with open("results.json", "w") as file:
                json.dump(
                    results,
                    file,
                    cls=AutoSklearnEncoder,
                    indent=2
                )

if __name__ == '__main__':
    AutoSklearnValidator.main()




