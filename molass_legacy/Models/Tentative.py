"""
    Tentative.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Models.ElutionModelUtils import get_xies_from_height_ratio

ENABLE_TAU_BOUND = False

class Model:
    def __init__(self, func=None, **kwargs):
        self.func = func
        self.name = "Model(%s)" % self.__class__.__name__
        self.delayed = kwargs.pop("delayed", False)
        self.tau_bound_ratio = get_setting("TAU_BOUND_RATIO")

    def get_name(self):
        return self.name

    def is_delayed(self):
        return self.delayed

    def is_traditional(self):
        return True

    def set_delayed_off(self):
        self.delayed = False

    def make_params(self, **kwargs):
        assert False

    def set_param_hint(self, pname, **kwargs):
        assert False

    def get_param_hints(self, pname):
        # remove this
        return None

    def fit(self, y, params, x=None, method=None):
        assert x is not None

        if method == "least_squares":
            # used in EmgPeak.py
            method = None

        def objective(p):
            try:
                y_ = self.func(x, *p)           # emga may return NaN
                chisqr = np.sum((y_ - y)**2)
            except:
                chisqr = np.inf
            return chisqr


        if ENABLE_TAU_BOUND:
            tau_bound_ratio = self.tau_bound_ratio

            cons = [
                    {'type': 'ineq', 'fun': lambda x:  x[2]*tau_bound_ratio - abs(x[3]) },  # sigma*tau_bound_ratio - abs(tau) >= 0
                    ]

            if method is None:
                method="Nelder-Mead"
        else:
            cons = None

        ret = minimize(objective, params, method=method, constraints=cons)

        chisqr = objective(ret.x)   # chisqr is referenced in ElutionDecomposer
        return Struct(params=ret.x, chisqr=chisqr)

    def eval(self, params, x=None):
        assert x is not None

        return self.func(x, *params)

    def __call__(self, x, params):
        return self.func(x, *params)

    def get_peaktop_xy(self, x, params):
        # this is a fallback. should be overridden if a faster implementation is available
        y = self.func(x, *params)
        m = np.argmax(y)
        return x[m], y[m]

    def get_proportions(self, x, params_array):
        areas = []
        for params in params_array:
            cy = self.func(x, *params)
            areas.append(np.sum(cy))
        return np.array(areas)/np.sum(areas)

    def get_range(self, x, y, alpha=0.1, max_y=None, debug=False):
        # task: unify the alpha value with the same meaning in other methods
        return get_xies_from_height_ratio(alpha, x, y, max_y=max_y, debug=debug)

    def guess_multiple(self, x, y, num_peaks, debug=False):
        # temporary support only for trditional models
        # task: add support for advanced models
        from molass_legacy.QuickAnalysis.ModeledPeaks import recognize_peaks  # placed here to avoid circular import
        assert self.get_name() in ["EGHA", "EMGA"] 
        return np.array(recognize_peaks(x, y, exact_num_peaks=num_peaks, affine=True, model=self, debug=debug))
    
    def guess_a_peak_with_prop(self, x, y, prop, **kwargs):
        raise NotImplementedError()     # should be overridden
    
    def guess_binary_peaks(self, x, y, p1, p2, **kwargs):
        raise NotImplementedError()     # should be overridden
    
    def guess_from_the_other(self, x, y, other_y, params_array):
        """
        This is not expected to work for advanced, i.e. non-traditional, models.
        """
        scale = np.sum(y)/np.sum(other_y)
        new_params_list = []
        for i, params in enumerate(params_array):
            new_params = params.copy()
            new_params[0] *= scale
            new_params_list.append(new_params)
        return np.array(new_params_list)