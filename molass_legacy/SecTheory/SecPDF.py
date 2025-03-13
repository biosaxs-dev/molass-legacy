"""
    SecTheory.SecPDF.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from .BasicModels import single_pore_pdf as c
from molass_legacy.KekLib.IntegrateUtils import complex_quadrature_vec

def compute_standard_wCD(N):
    # extracted from .cf2DistFFT.py
    xMin = 0
    xMax = N
    xRange = xMax - xMin
    dt  = 2*np.pi / xRange
    # dt = 1/xRange
    k   = np.arange(N, dtype=complex)     # np.complex is deprecated, or use np.complex128
    w   = (k - N/2 + 0.5) * dt
    A   = xMin
    B   = xMax
    # dx  = (B-A)/N
    c   = (-1)**(A*(N-1)/(B-A))/(B-A)
    # print("A, B, N, dx, c=", A, B, N, dx, c)
    C = c * (-1)**((1-1/N)*k)
    D = (-1)**(-2*(A/(B-A))*k)     # k must be complex, see https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
    return w, C, D

class FftInvPdf:
    def __init__(self, cf):
        self.cf = cf
        self.N = N = 1024
        self.w, self.C, self.D = compute_standard_wCD(N)

    def __call__(self, t, *params):
        N = self.N
        cft = self.cf(self.w[N//2:], *params)
        cft = np.concatenate([cft[::-1].conj(), cft])
        pdfFFT = np.max([np.zeros(N), (self.C*np.fft.fft(self.D*cft)).real], axis=0)
        spline = UnivariateSpline(np.arange(N), pdfFFT, s=0)
        return spline(t)

class FftInvImpl:
    def __init__(self):
        self.N = N = 1024
        self.w, self.C, self.D = compute_standard_wCD(N)
    
    def get_w(self):
        return self.w

    def compute(self, t, z):
        N = self.N
        z_ = z[N//2:]
        z_ = np.concatenate([z_[::-1].conj(), z_])
        pdfFFT = np.max([np.zeros(N), (self.C*np.fft.fft(self.D*z_)).real], axis=0)
        spline = UnivariateSpline(np.arange(N), pdfFFT, s=0)
        return spline(t)
    
class BidirectFft:
    def __init__(self, x, N=1024):
        self.x = x
        self.N = N 
        self.w, self.C, self.D = compute_standard_wCD(N)
    
    def get_w(self):
        return self.w

    def compute_z(self, y):
        spline = UnivariateSpline(self.x, y, s=0)
        return complex_quadrature_vec(lambda x_: np.exp(1j*self.w*x_)*spline(x_), self.x[0], self.x[-1])[0]

    def compute_y(self, z):
        N = self.N
        z_ = z[N//2:]
        z_ = np.concatenate([z_[::-1].conj(), z_])
        pdfFFT = np.max([np.zeros(N), (self.C*np.fft.fft(self.D*z_)).real], axis=0)
        spline = UnivariateSpline(np.arange(N), pdfFFT, s=0)
        return spline(self.x)