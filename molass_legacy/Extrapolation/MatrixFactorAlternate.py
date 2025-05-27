# coding: utf-8
"""
    MatrixFactorAlternate.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from molass_legacy.Models.ElutionCurveModels import emg
import molass_legacy.KekLib.DebugPlot as plt

def design_figure():
    fig = plt.figure()
    ax = fig.gca()
    x = np.arange(100)
    y1 = emg(x, h=1, mu=50, sigma=5)

    spline = UnivariateSpline([0, 20, 40, 60, 80, 99], [0.2, 0.11, 0.1, 0.03, 0.02, -0.1])
    # y2 = -0.003*x + 0.2
    y2 = spline(x)

    ax.set_title("Decomposition of Concentration", fontsize=16)
    ax.plot(x, y1, ':', linewidth=3, label='nomal elution element')
    ax.plot(x, y2, ':', linewidth=3, label='baseline drift')
    ax.plot(x, y1+y2, ':', color='red', linewidth=3, label='total elution')

    ax.legend(fontsize=16)
    fig.tight_layout()
    plt.show()

class MatrixFactorAlternate:
    def __init__(self, sd):
        self.q = sd.intensity_array[0,:,0]
        self.data = sd.intensity_array[:,:,1].T
        self.e_curve = sd.xray_curve

        peak_i = self.e_curve.primary_peak_i
        xslice = slice(0,100)
        yslice = slice(peak_i-50, peak_i+50)
        y = np.arange(self.data.shape[1])[yslice]

        x = self.q[xslice]
        xx, yy = np.meshgrid( x, y )
        i = np.arange(xslice.start, xslice.stop)
        j = np.arange(yslice.start, yslice.stop)
        ii, jj = np.meshgrid( i, j )

        zz = self.data[ii, jj]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.3 )

        plt.show()

