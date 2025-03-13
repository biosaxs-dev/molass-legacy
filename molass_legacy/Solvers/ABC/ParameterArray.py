"""
    Solvers.ABC.ParameterArray.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from pyabc import Parameter

MAX_NUM_PARAMS = 100

class ParameterArray(Parameter):
    def __init__(self, rv_array):
        assert len(rv_array) <= MAX_NUM_PARAMS
        self.rv_array = np.asarray(rv_array)
        self.updated = False

    def update_dict(self):
        for i, v in enumerate(self.rv_array):
            self["p%02d" % i] = v
        self.updated = True

    def __getitem__(self, key):
        if type(key) is str:
            if not self.updated:
                self.update_dict()
            return Parameter.__getitem__(self, key)
        else:
            return self.rv_array[key]

    def get(self, key):
        if not self.updated:
            self.update_dict()
        return Parameter.get(self, key)

    def copy(self) -> "ParameterArray":
        """
        Copy the parameter.
        """
        return ParameterArray(self.rv_array)

    def get_values(self):
        return self.rv_array

def parameter_to_values(df):
    try:
        return df.get_values()
    except:
        print("dataframe_to_values: type(df)=", type(df))
        values = df.values
        print("type(values), values=", type(values), values)
        return values