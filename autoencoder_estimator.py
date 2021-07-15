import mlflow
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
        X_ = X.values

        train_data, test_data = train_test_split(
            X_, test_size=0.2, shuffle=False
        )
        train_data = tf.cast(train_data, tf.float32)
        test_data = tf.cast(test_data, tf.float32)

        self.autoencoder = AutoEncoder()
        self.autoencoder.compile(optimizer='adam', loss='mae')

        self.autoencoder.fit(train_data, train_data, 
          epochs=self.epochs, 
          batch_size=self.batch_size,
          validation_data=(test_data, test_data),
          shuffle=True)

        reconstructions = self.autoencoder.predict(X_)
        train_loss = tf.keras.losses.mae(reconstructions, X_)
        self.threshold = np.mean(train_loss) + np.std(train_loss)

        return self
    
    def transform(self, X, y=None):
        return self

    def predict(self, X):
        X_ = X.values

        print(X_.shape)
        print(type(X_))

        reconstructions = self.autoencoder(X_)
        losses = tf.keras.eses.mae(reconstructions, X_)

        return tf.math.less(self.threshold, losses)
