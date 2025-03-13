"""
    Solvers.ABC.ConstrainedPrior.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.special import gamma
from importlib import reload
from pyabc import RV, DistributionBase, Parameter
import Solvers.ABC.ParameterUtils
reload(Solvers.ABC.ParameterUtils)
from Solvers.ABC.ParameterUtils import vector_to_parameter, parameter_to_vector

MAX_NUM_PARAMS = 100
PDF_FACTOR_SCALE = 1e-1     # experimental. to do this correctly, see also numpy.isclose

class ConstrainedPrior(DistributionBase):
    def __init__(self, shape, lower, upper, offset=1):
        self.num_all_params = len(lower)
        assert self.num_all_params <= MAX_NUM_PARAMS
        num_components, model_num_params = shape
        rv_list = []
        for lb, ub in zip(lower, upper):
            # rv_list.append(RV("uniform", lb, ub))
            loc = (lb + ub)/2
            scale = (ub - lb)*1/3
            rv_list.append(RV("norm", loc=loc, scale=scale))
        self.rv_list = rv_list
        self.ordering_slice = np.array(offset + np.arange(num_components) * model_num_params, dtype=int)
        n = self.num_all_params
        print("ordering_slice=", self.ordering_slice)
        self.volume_ratio = np.power(np.pi, n/2)/gamma(n/2 + 1)*np.power(2.0, -n)   # 4.13e-19 for n=37
        self.pdf_factor = 1/self.volume_ratio

    def _unconstraned_rvs(self):
        ret_values = np.zeros(len(self.rv_list))
        for i, p in enumerate(self.rv_list):
            ret_values[i] = p.rvs()
        return ret_values
    
    def rvs(self, *args, **kwargs):
        while True:
            values = self._unconstraned_rvs()
            ordering_values = values[self.ordering_slice]
            if np.min(np.diff(ordering_values)) > 0:
                return vector_to_parameter(values)

    def pdf(self, x):
        # values = np.array([x[PARAM_KEY_FORMAT % k] for k in range(self.num_all_params)])
        values = parameter_to_vector(x)
        ordering_values = values[self.ordering_slice]
        if np.min(np.diff(ordering_values)) <= 0:
            return 0.0
        # print("values=", values)
        # print("ordering_values=", ordering_values)
        # return np.product([rv.pdf(p) for rv, p in zip(self.rv_list, values)])
        p = np.product([rv.pdf(p) for rv, p in zip(self.rv_list, values)])
        # print("p=", p)
        return p * self.pdf_factor