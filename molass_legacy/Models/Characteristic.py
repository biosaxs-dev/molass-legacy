"""
    Characteristic.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.SecTheory.SecPDF import compute_standard_wCD

class CfSpace:
    def __init__(self, N=1024, use_np_fft=True):
        self.use_np_fft = use_np_fft
        if use_np_fft:
            self.w = np.fft.fftfreq(N)
        else:
            w, C, D = compute_standard_wCD(N)
            self.w = w
            self.C = C
            self.D = D

    def get_w(self):
        return self.w

    def compute_cf(self, x, y):
        self.area = np.sum(y)
        y_ = y/self.area
        if self.use_np_fft:
            cft = np.fft.fft(y_, n=len(self.w))
        else:
            cft = []
            for w_ in self.w:
                cft.append(np.sum(np.exp(1j*w_*x)*y_))
            cft = np.array(cft)
        return cft
