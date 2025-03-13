"""
    ModelEvaluator.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from molass_legacy.SerialAnalyzer.ElutionCurve import ANALYSIS_RANGE_RATIO

class ModelEvaluator:
    def __init__(self, model, params, sign=1, accepts_real_x=False, debug=False):
        self.model = model
        self.model_name = model.get_name()
        if model.is_traditional():
            self.param_values = params[0:5].copy()  # why 0:5
        else:
            self.param_values = params.copy()

        if len(params) > 5:
            # ('const_a', <Parameter 'const_a', value=14.635374905168955 +/- 0.565, bounds=[0:inf], expr='sigma*1.5 - abs(a)'>)])
            # self.const_a = params[5]
            pass

        self.sign = sign
        self.accepts_real_x = accepts_real_x    # for temporary fix under V1PreviewAdapter
        self.func = model.func

    def get_model_name( self ):
        return self.model_name

    def get_func(self):
        return self.func

    def set_new_params(self, param_values):
        self.param_values = param_values

    def get_param_value(self, k):
        return self.param_values[k]

    def get_param_values(self):
        return self.param_values[0:4]

    def get_all_param_values(self):
        return self.param_values

    def get_all_params_string(self):
        values = self.param_values.copy()
        values[0] *= self.sign      # TODO: include sign to h instead of multiplying here
        return self.model.get_params_string(values)

    def get_range_params( self, x, ratio=ANALYSIS_RANGE_RATIO ):
        """
        TODO: consider unifying this logic with EmgPeak.get_model_x_from_ratio(...)
        """
        y = self.__call__(x)
        top_x = np.argmax(y)
        top_y = y[top_x]
        foot_y = top_y * ratio
        i = bisect_right(y[0:top_x], foot_y)
        y_ = y[top_x:]
        j = bisect_right(y_[::-1], foot_y)
        return i, top_x, len(x) - 1 - j

    def get_model_def_expr( self ):
        return self.model.get_def_expr(self.lmfit_params)

    def update_param( self, k, value ):
        self.param_values[k] = value

    def __call__(self, x, debug=False):
        # 
        if debug:
            print("ModelEvaluator.__call__: param_values==", self.param_values)
        if np.isscalar(x):
            return self.func(np.array([x]), *self.param_values)[0] * self.sign
        else:
            return self.func(x, *self.param_values) * self.sign

    def __repr__( self ):
        return "%s(%s(),%s)" % (self.__class__.__qualname__, self.get_model_name(), repr(self.param_values))

    def x_from_height_ratio(self, ecurve, ratio):
        return self.model.x_from_height_ratio(ecurve, ratio, self.param_values)
