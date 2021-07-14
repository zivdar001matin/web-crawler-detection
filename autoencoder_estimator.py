import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split

class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(27, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderEstimator(BaseEstimator):
    def __init__(self, epochs=10, batch_size=512) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])
