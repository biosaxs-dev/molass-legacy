"""
    DualEvaluator.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
import copy
import numpy as np
from bisect import bisect_right
from molass_legacy.SerialAnalyzer.ElutionCurve import PEAK_INFO_FIND_RATIO
from ModelEvaluator import ModelEvaluator

class DualEvaluator(ModelEvaluator):
    def __init__(self, model, param_values, debug=False):
        self.model = model
        self.model_name = model.get_name()
        self.func = model.func
        if len(param_values) != 5:
            print("DualEvaluator: param_values=", param_values)
        assert len(param_values) == 5
        self.param_values = param_values
        self.sign = 1   # no need for negative sign?
        self.accepts_real_x = False     # for temporary fix under V1PreviewAdapter
