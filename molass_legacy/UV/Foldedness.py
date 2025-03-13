"""
    UV.Foldedness.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
from bisect import bisect_right

class Foldedness:
    def __init__(self, wv):
        self.wv = wv
        self.indeces = [bisect_right(wv, w) for w in self.get_wavelengths()]

    def get_wavelengths(self):
        return [280, 275, 258]

    def get_plotcolors(self):
        return ["red", "cyan", "gray"]

    def compute(self, pv):
        i = self.indeces
        return pv[i[0]]/pv[i[1]] + pv[i[0]]/pv[i[2]]

if __name__ == '__main__':
    import numpy as np
    wv = np.arange(250,300)
    f = Foldedness(wv)
    print(f.compute(np.arange(50)))
