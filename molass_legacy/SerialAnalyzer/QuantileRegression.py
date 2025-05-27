# coding: utf-8
"""
    QuantileRegression.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
# from QuantileRegressionKeras import Model
from QuantileRegressionSklearn import Model
import molass_legacy.KekLib.DebugPlot as plt

class QuantileRegression:
    def __init__(self, sd):
        array = sd.intensity_array

        N = 11

        q = array[0,:,0]
        qq = np.hstack([q]*N)

        i = 340
        curve_y = array[i,:,1]
        H = N//2
        yy = np.hstack([ array[k,:,1] for k in range(i-H, i+H+1) ])

        model = Model()
        model.fit(qq, yy)
        learned_y = model.predict(q)

        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(sd.xray_curve.y, label='smp')

        for j in range(0, array.shape[1], 100):
            ax1.plot(array[:,j,1], label='j=%d' % j)

        ax2.plot(q, curve_y, label='i=%d' % i)
        ax2.plot(q, learned_y, label='learned')

        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.show()
