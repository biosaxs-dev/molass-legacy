# coding: utf-8
"""
    SaxsDetector.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .ReciprocalData import ReciprocalData

USE_ZERNIKE_COEF = False

class SaxsDetector:
    def __init__(self, fig, ax, data, q, in_y):
        ax.set_title("Detector Image", fontsize=20)
        ax.set_axis_off()

        if USE_ZERNIKE_COEF:
            from .OurOptics import ZernikeCoefficient
            Z = ZernikeCoefficient(Z11=1)
            Z.zernikemap(fig=fig, ax=ax, cmap=cm.plasma, label=False)
            return

        rdata = ReciprocalData(data.shape)
        F = rdata.get_reciprocal(data)
        self.curve_y = y = rdata.get_scattering_curve(q, F)
        self.draw_detector_image(fig, ax, q, y, in_y)

    def compute_zernike_coefficients(self, data):
        # TODO
        pass


    def draw_detector_image(self, fig, ax, q, y, in_y):
        assert len(q) == len(y)
        n = len(y)
        qmax = q[-1]
        theta = np.linspace(0, 2*np.pi, 400)
        rho = np.linspace(0, qmax, 400)
        u,r = np.meshgrid(theta,rho)
        X = r*np.cos(u)
        Y = r*np.sin(u)
        # r -> y
        scale = 1 if in_y is None else in_y[0]/y[0]
        interp = interpolate.interp1d(q, y*scale, kind='cubic', fill_value="extrapolate")
        Z = interp(r)
        im = ax.pcolormesh(X, Y, Z, cmap=cm.plasma)
        fig.colorbar(im, ax=ax)
        ax.set_aspect('equal', 'datalim')
