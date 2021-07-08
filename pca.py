from pprint import pprint
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow
from fetch_data import fetch_logged_data
from log_transformer import LogTransformer

def main():
    # enable autologging
    mlflow.sklearn.autolog()

    # prepare training data
    X = pd.read_csv('datasets/unprocessed.csv')
    X = X.drop(X.loc[X['ip'] == '-'].index)
    X = X.reset_index(drop=True)

    # train a model
    pipe = Pipeline([("transform", LogTransformer()), ("pca", PCA())])
    with mlflow.start_run() as run:
        pipe.fit_transform(X)
        print("Logged data and model in run: {}".format(run.info.run_id))

    # show logged data
    for key, data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)


if __name__ == "__main__":
    main()