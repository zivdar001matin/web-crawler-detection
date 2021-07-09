import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

class PCAEstimator(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        X_ = X.values

        self.pca = PCA()

        """ Projection """
        X_transformed = self.pca.fit_transform(X_)

        """ Reconstruct """
        X_reconstructed = self.pca.inverse_transform(X_transformed)

        loss = np.sum((X_ - X_reconstructed)**2, axis=1)

        # Consider 10% of data as anomaly
        self.threshold = np.percentile(loss, 90)

        return self

    
    def transform(self, X, y=None):
        return self

    def predict(self, X):
        X_ = X.values

        X_reconstructed = self.pca.inverse_transform(self.pca.transform(X_))
        loss = np.sum((X_ - X_reconstructed)**2, axis=1)

        return loss > self.threshold