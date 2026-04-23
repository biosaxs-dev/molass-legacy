"""
    ColumnInterp.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

class ColumnInterp:
    def __init__(self, D, j0=None):
        # add same columns to both sides
        D_ = np.vstack([D[:,0], D.T, D[:,-1]]).T
        # Store in Fortran (column-major) order so that D_[:, j] accesses a
        # single contiguous 400-float chunk instead of scattered row-major
        # memory.  This converts column extraction from a cache-miss-heavy
        # scatter read to a sequential read, which is much faster.
        self.D_ = np.asfortranarray(D_)
        self.size = D.shape[1]
        self.j0 = j0

    def safe_index(self, i):
        # Replaces the 4-array np.max/np.min/np.zeros/np.ones chain with a
        # single np.clip call, eliminating three temporary array allocations.
        return np.clip(np.asarray(i, dtype=int), 0, self.size + 1)

    def __call__(self, j):
        if not isinstance(j, np.ndarray):
            j = np.array(j)
        if self.j0 is None:
            self.j0 = int(j[0])
        j_ = j - self.j0 + 1
        # Replace np.divmod(j_, np.ones(len(j))) — avoids allocating a ones
        # array just to divide by 1.  j_ >= 1 always (j_ = j - j0 + 1 and
        # j[0] == j0), so integer truncation equals floor.
        col = j_.astype(int)
        res = j_ - col
        col1 = np.clip(col,     0, self.size + 1)
        col2 = np.clip(col + 1, 0, self.size + 1)
        # D_ is Fortran-order → each D_[:, k] is a contiguous 400-float read.
        # Use in-place operations to halve the number of temporary (rows×n)
        # allocations: 2 (from the two takes) instead of the original 5.
        tmp1 = self.D_[:, col1]
        tmp2 = self.D_[:, col2]
        tmp1 *= (1 - res)
        tmp2 *= res
        tmp1 += tmp2
        return tmp1
