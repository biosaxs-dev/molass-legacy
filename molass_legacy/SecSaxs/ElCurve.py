"""
    SecSaxs.ElCurve.py

    1) temporary successor to ElutionCurve
    2) x starts with a value at the slice start when trimmed

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import logging
import os
import numpy as np
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt

SEDIMENTATION_LIMIT = 0.03
END_DIFF_WIDTH = 50
END_DIFF_MIN_HEIGHT = SEDIMENTATION_LIMIT/2
RELIABLE_SIGMA_RATIO = 2
RELIABLE_ALPHA = 0.03
USE_RELIABLE_ALPHA = True

class ElCurve:
    def __init__(self, x, y, v1_curve=None, debug=False):
        self.logger = logging.getLogger(__name__)
        self.x = x
        self.y = y
        if v1_curve is None:
            from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
            v1_curve = ElutionCurve(y, x=x)
        self.peak_top_x = v1_curve.peak_top_x + x[0]
        self.peak_info = [[x[0] + j for j in info] for info in v1_curve.peak_info]
        self.boundaries = [x[0] + j for j in v1_curve.boundaries]
        self.emg_peaks = [epeak.shift_copy(x[0], x[-1]+1) for epeak in v1_curve.emg_peaks]       
        self.sy = v1_curve.sy
        print("len(x), len(y), len(sy):", len(x), len(y), len(self.sy))
        self.spline = UnivariateSpline(x, self.sy, s=0, ext=3)
        self.primary_peak_no = v1_curve.primary_peak_no
        if v1_curve is None or True:
            j = np.argmax(y)
            self.max_j = j
            self.max_x = x[j]
            self.max_y = y[j]
        else:
            self.max_x = v1_curve.x[v1_curve.max_x]
            self.max_y = v1_curve.max_y
        self.mean = None
        self.variance = None
        self.sigma = None
        self.end_slices = None
        self.primary_peak_x = self.max_x    # temporary fix for 20250212/SG
        self.peak_region_sigma = None       # temporary fix for 20250212/SG

        if debug:
            with plt.Dp():
                fig, axes = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("ElCurve.__init__")
                for ax, curve in zip(axes, [v1_curve, self]):
                    x = curve.x
                    y = curve.y
                    ax.plot(x, y)
                    for px in curve.peak_top_x:
                        ax.plot(px, curve.spline(px), "o", color="red")
                fig.tight_layout()
                plt.show()

    def get_xy(self):
        # for forward compatibility
        return self.x, self.y

    def get_primarypeak_x(self):
        return self.max_x

    def get_primarypeak_i(self):
        return int(round(self.max_x - self.x[0]))
    
    def compute_moments(self):
        if self.mean is None:
            x = self.x
            y = np.zeros(len(x))

            # self.y can be negative as in 20200123_1
            positive_y = self.y > 0
            y[positive_y] = self.y[positive_y]

            sum_y = np.sum(y)
            mean  = np.sum(x*y)/sum_y
            variance = np.sum(x**2*y)/sum_y - mean**2
            self.mean = mean
            self.variance = variance
            self.sigma = np.sqrt(variance)

        return self.mean, self.variance

    def get_sigma(self):
        if self.sigma is None:
            self.compute_moments()

        return self.sigma

    def get_emg_peaks(self, **kwargs):
        return self.emg_peaks

    def get_peak_region_sigma(self):
        if self.peak_region_sigma is None:
            emg_peaks = self.get_emg_peaks()
            h1, m1, s1, t1 = emg_peaks[0].get_params()
            h2, m2, s2, t2 = emg_peaks[-1].get_params()
            peak_region = (max(0, m1-3*s1), min((len(self.x)), m2+3*s2))
            self.peak_region_sigma = peak_region
        return self.peak_region_sigma

    def get_end_y(self, slice_):
        y_ = self.y[slice_]
        h = abs(y_[-1] - y_[0])
        ok = False
        if h/self.max_y < END_DIFF_MIN_HEIGHT:
            ok = True
        else:
            std = np.std(y_)
            if std/self.max_y < END_DIFF_MIN_HEIGHT:
                ok = True
        if ok:
            self.end_slices.append(slice_)
            return np.average(y_)

        # in this case, slice_ must be narrowed as obsesrved in 20181204
        if slice_.start == 0:
            return self.get_end_y(slice(0, slice_.stop//2))
        else:
            return self.get_end_y(slice(slice_.start//2, None))

    def compute_sedimentation_rate(self, debug=False):
        emg_peaks = self.get_emg_peaks()
        self.end_slices = []
        start_y = self.get_end_y(slice(0, END_DIFF_WIDTH))
        end_y = self.get_end_y(slice(-END_DIFF_WIDTH, None))
        sedrate = (end_y - start_y)/(self.max_y - start_y)

        slice1 = self.end_slices[0]
        width1 = slice1.stop

        slice2 = self.end_slices[1]
        width2 = abs(slice2.start)
        start2 = len(self.x) + slice2.start

        num_emg_peaks = len(emg_peaks)

        if num_emg_peaks > 0:
            # to reduce the calulation
            indeces = set([0, num_emg_peaks-1])

            lim_x_pairs = []
            for i in indeces:
                if USE_RELIABLE_ALPHA:
                    pair = emg_peaks[i].get_model_x_from_ratio(RELIABLE_ALPHA)
                else:
                    pair = emg_peaks[i].get_sigma_points(RELIABLE_SIGMA_RATIO)
                lim_x_pairs.append(pair)
            sigma_x1 = lim_x_pairs[0][0]
            sigma_x2 = lim_x_pairs[-1][1]
            reliable = (width1 >= END_DIFF_WIDTH//8 and slice1.stop < sigma_x1) and (width2 >= END_DIFF_WIDTH//8 and sigma_x2 < start2)
            if debug:
                print(self.end_slices, sigma_x1, sigma_x2, start2)
                print(width1, width2)

                x = self.x
                y = self.y

                plt.push()
                fig, ax = plt.subplots()
                ax.plot(x, y, label='data')
                for slice_ in [slice(0,END_DIFF_WIDTH), slice(-END_DIFF_WIDTH, None)]:
                    ax.plot(x[slice_], y[slice_], 'o', color='yellow')

                for peak in emg_peaks:
                    print(self.j0, peak, peak.get_sigma_points(RELIABLE_SIGMA_RATIO))

                for k in set([0, len(emg_peaks)-1]):
                    peak = emg_peaks[k]
                    ax.plot(x, peak.get_model_y(x), label='model')

                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                for sx in [sigma_x1, sigma_x2]:
                    ax.plot([sx, sx], [ymin, ymax], ':', color='gray')

                ax.legend()
                fig.tight_layout()
                plt.show()
                plt.pop()
        else:
            # temporary fix for pH6 to avoid a bug with the revised EmgPeak
            reliable = False

        self.logger.info("computed sedimentation rate as %.3g which is%s reliable", sedrate, "" if reliable else " not")
        return sedrate, reliable

    def get_end_slices(self):
        if self.end_slices is None:
            self.compute_sedimentation_rate()
        return self.end_slices
