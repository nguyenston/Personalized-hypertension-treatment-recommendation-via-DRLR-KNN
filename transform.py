import sys
sys.path.append('/home/yeping/HTNProject/Personalized-hypertension-treatment-recommendation-via-DRLR-KNN')
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from dist_robust_regress import DistributionallyRobustRegressor


class OLSTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scale_factor_ = None

    def fit(self, x, y):
        self._fit(x, y)
        return self

    def _fit(self, x, y):
        ols = LinearRegression()
        ols.fit(x, y)

        self.scale_factor_ = ols.coef_
        self.scale_factor_ = np.reshape(np.abs(self.scale_factor_), [1, -1])

    def transform(self, X):
        X_reshape = np.multiply(X, self.scale_factor_)
        return X_reshape


class DRLRTransformer(BaseEstimator, TransformerMixin):
    # WARNING: k parameter for debug purposes, to be removed later
    def __init__(self, reg_l2=0.1, reg_l1=0.1, k=0, solver="scipy"):
        self.k = k
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.solver = solver

        self.scale_factor_ = None

    def fit(self, x, y):
        print("L2 = {}\t K = {}".format(self.reg_l2, self.k))
        self._fit(x, y)
        return self

    def _fit(self, x, y):
        drlr = DistributionallyRobustRegressor(reg_l2=self.reg_l2, reg_l1=self.reg_l1, solver=self.solver)
        drlr.fit(x, y)

        self.scale_factor_ = drlr.coef_
        self.scale_factor_ = np.reshape(np.abs(self.scale_factor_), [1, -1])

    def transform(self, X):
        return np.multiply(X, self.scale_factor_)
