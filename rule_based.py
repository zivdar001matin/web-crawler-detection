import numpy as np
from pandas.api.types import is_sparse
from numpy.lib.function_base import select
from sklearn.base import BaseEstimator

class RuleBasedEstimator(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X

    def predict(self, X):
        X_ = X.copy()

        # Rule No.1
        X_['rule_based_predict'] = X_['device_brand_Spider'].values

        if is_sparse(X_['rule_based_predict']):
            rule_based_predict = X_['rule_based_predict'].sparse.to_dense().values
        else:
            rule_based_predict = X_['rule_based_predict'].values

        if is_sparse(X_['pca_predict']):
            pca_predict = X_['pca_predict'].sparse.to_dense().values
        else:
            pca_predict = X_['pca_predict'].values

        predict = np.logical_or(rule_based_predict, pca_predict)

        return predict