# coding: utf-8
"""
    OurOptics.py

    custom class of opticspy

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from opticspy.zernike import Coefficient
from opticspy import interferometer_zenike as __interferometer__

class ZernikeCoefficient(Coefficient):
    def __init__(self, **kwargs):
        Coefficient.__init__(self, **kwargs)

    def zernikemap(self, fig=None, ax=None, cmap=None, label=True):

        theta = np.linspace(0, 2*np.pi, 400)
        rho = np.linspace(0, 1, 400)
        [u,r] = np.meshgrid(theta,rho)
        X = r*np.cos(u)
        Y = r*np.sin(u)
        Z = __interferometer__.__zernikepolar__(self.__coefficients__,r,u)

        arg_ax = ax
        if arg_ax is None:
            fig = plt.figure(figsize=(12, 8), dpi=80)
            ax = fig.gca()

        if cmap is None:
            cmap = cm.RdYlGn
        im = ax.pcolormesh(X, Y, Z, cmap=cmap)

        if label:
            plt.title('Zernike Polynomials Surface Heat Map',fontsize=18)
            ax.set_xlabel(self.listcoefficient()[1],fontsize=18)

        fig.colorbar(im, ax=ax)
        ax.set_aspect('equal', 'datalim')

        if arg_ax is None:
            plt.show()
