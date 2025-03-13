"""
    QuickAnalysis.ParammedModel.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np

class ParammedModel:
    def __init__(self, model_params):
        self.model_params = model_params

    def append(self, model_spec):
        self.model_params.append(model_spec)

    def __call__(self, x):
        y = np.zeros(len(x))

        for model, params in self.model_params:
            y += model.eval(params, x=x)

        return y
