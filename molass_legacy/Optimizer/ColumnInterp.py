"""
    ColumnInterp.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

class ColumnInterp:
    def __init__(self, D, j0=None):
        # add same columns to both sides
        self.D_ = np.vstack([D[:,0], D.T, D[:,-1]]).T
        self.size = D.shape[1]
        self.j0 = j0

    def safe_index(self, i):
        i_ = np.array(i, dtype=int)
        return np.max([np.zeros(i.shape, dtype=int),
                        np.min([np.ones(i.shape, dtype=int)*(self.size+1), i_], axis=0)], axis=0)

    def __call__(self, j):
        if type(j) != np.ndarray:
            j = np.array(j)
        if self.j0 is None:
            self.j0 = int(j[0])
        j_ = j - self.j0 + 1
        col, res = np.divmod(j_, np.ones(len(j)))
        col1 = self.safe_index(col)
        col2 = self.safe_index(col1+1)
        return self.D_[:,col1]*(1-res) + self.D_[:,col2]*res
