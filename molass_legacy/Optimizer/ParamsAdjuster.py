# coding: utf-8
"""
    ParamsAdjuster.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""

class ParamsAdjuster:
    def __init__(self, dsets, n_components, init_params):
        self.dsets = dsets
        self.n_components = n_components
        self.init_params = init_params

    def get_params_for(self, n):
        num_diff = n - self.self.n_components
        if num_diff == 0:
            ret_params = self.init_params
        elif num_diff > 0:
            ret_params = self.add_new_components(num_diff)
        else:
            ret_params = self.remove_extra_components(-num_diff)
        return ret_params

    def add_new_components(self, num):
        pass

    def remove_extra_components(self, num):
        pass
