"""
    ModelParams.BaseSliderInfo.py

    Copyright (c) 20224, SAXS Team, KEK-PF
"""
import numpy as np

class BaseSliderInfo:
    def __init__(self,
                 cmpparam_names=[],
                 cmpparam_indeces=[],
                 whlparam_names=[],
                 whlparam_indeces=[]
                 ): 
        self.cmpparam_names = cmpparam_names
        self.cmpparam_indeces = np.asarray(cmpparam_indeces, dtype=int)
        self.whlparam_names = whlparam_names
        self.whlparam_indeces = np.asarray(whlparam_indeces, dtype=int)

    def get_component_indeces(self, i):
        return self.cmpparam_indeces[i,:]