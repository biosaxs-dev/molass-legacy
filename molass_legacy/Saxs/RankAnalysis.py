# coding: utf-8
"""
    RankAnalysis.py

    custom class of opticspy

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from SvdDenoise import get_denoised_data
import molass_legacy.KekLib.DebugPlot as plt

class RankAnalysis:
    def __init__(self, data):

        scores = []
        for rank in range(1, 9):
            D = get_denoised_data(data, rank=rank)
            scores.append(np.linalg.norm(D - data))

        plt.push()
        plt.plot(scores, '-o')
        plt.show()
        plt.pop()
