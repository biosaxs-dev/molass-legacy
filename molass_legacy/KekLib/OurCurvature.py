# coding: utf-8
"""
    OurCurvature.py

    Copyright (c) 2016-2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
from scipy.interpolate      import UnivariateSpline

def curvature_curve( x, y, spline=None ):

    if spline is None:
        fy = UnivariateSpline( x, y, k=3, s=0 )
    else:
        fy = spline

    yˈ  = fy.derivative(1)(x)
    yˈˈ = fy.derivative(2)(x)

    cy = yˈˈ / np.power( 1 + yˈ**2, 3/2 )

    return cy
