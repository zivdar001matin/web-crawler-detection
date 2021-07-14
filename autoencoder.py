import mlflow
import argparse
import sys
import pandas as pd
import mlflow.tensorflow

from pprint import pprint
from fetch_data import fetch_logged_data
from log_transformer import LogTransformer
from autoencoder_estimator import AutoEncoderEstimator
from sklearn.pipeline import Pipeline

DATA_URL = "datasets/unprocessed.csv"

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.sklearn.autolog()
mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=1000, type=int, help="number of epochs")
parser.add_argument("--batch_size", default=100, type=int, help="batch size")

def main(argv):
    args = parser.parse_args(argv[1:])
        
    # prepare training data
    data = pd.read_csv(DATA_URL)
    data = data.drop(data.loc[data['ip'] == '-'].index)
    data = data.reset_index(drop=True)

    # train a model
    pipe = Pipeline(
    [
        ("transform", LogTransformer()),
        ("autoencoder", AutoEncoderEstimator())
    ]
    , verbose=True)

    with mlflow.start_run() as run:
        pipe.fit_transform(data)
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
    main(sys.argv)