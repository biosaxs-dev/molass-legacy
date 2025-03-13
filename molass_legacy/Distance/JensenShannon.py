"""
    Distance.JensenShannon.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.spatial import distance
from molass_legacy.Trimming.Sigmoid import sigmoid, sigmoid_inv
import molass_legacy.KekLib.DebugPlot as plt

def deformed_jsd(y1, y2, b=None, debug=False):
    jsd = 0.5*distance.jensenshannon(y1, y2)
    a = np.percentile(y1, 95)
    if b is None:
        b = np.percentile(y2, 95)
    area_ratio = abs(a - b)/(a + b)
    ret_score = max(-100, 5*sigmoid_inv(max(jsd, area_ratio), 1, 0.01, 10.0, 0) + 0.5)

    if debug:
        print("jsd=", jsd, "area_ratio=", area_ratio, "ret_score=", ret_score)
        x = np.arange(len(y2))
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("stochastic_measure debug")
            ax.plot(x, y1)
            ax.plot(x, y2)
            fig.tight_layout()
            plt.show()

    return ret_score
