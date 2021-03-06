from pprint import pprint
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

import mlflow
from fetch_data import fetch_logged_data
from log_transformer import LogTransformer
from pca_estimator import PCAEstimator
from rule_based import RuleBasedEstimator

def main():
    # enable autologging
    mlflow.sklearn.autolog()

    # prepare training data
    X = pd.read_csv('datasets/unprocessed.csv')
    X = X.drop(X.loc[X['ip'] == '-'].index)
    X = X.reset_index(drop=True)

    # train a model
    pipe = Pipeline(
        [
            ("transform", LogTransformer()),
            ("pca", PCAEstimator()),
            ("rule_based", RuleBasedEstimator())
        ]
        , verbose=True)
    with mlflow.start_run() as run:
        pipe.fit_transform(X)
        print("Logged data and model in run: {}".format(run.info.run_id))

        # # For loading model in the old way (load from pickle)
        # mlflow.sklearn.log_model(pipe, 'model')

        # # Save training data
        # mlflow.log_artifact('datasets/unprocessed.csv')

    # show logged data
    for key, data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)


if __name__ == "__main__":
    main()