# coding: utf-8
"""
    QuantileRegressionSklearn.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class Model:
    def __init__(self):
        self.model = GradientBoostingRegressor(loss='quantile', alpha=0.5,
                                n_estimators=10, max_depth=3,
                                learning_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)

    def fit(self, x, y):
        X = np.atleast_2d(x).T
        self.model.fit(X, y)

    def predict(self, x):
        X = np.atleast_2d(x).T
        return self.model.predict(X)
