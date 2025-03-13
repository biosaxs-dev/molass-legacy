"""
    PairedDataSets.py.

    Copyright (c) 2020, 2025, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
from .ProbDensityUtils import generate_sample_data
from .GroupingMatrix import num_to_char

def get_cut_num_peaks(ecurve):
    """
    be aware that it is assumed here
    that this ecurve is not yet cut,
    i.e. ecurve.x retains the original elution numbers.
    """
    if ecurve is None:
        # as in 20190309_3
        num_peaks = 1
    else:
        peaks = ecurve.get_major_peak_info(add_j0=True)
        qmm_window_slice = get_setting('qmm_window_slice')
        print('qmm_window_slice=', qmm_window_slice, 'peaks=', peaks)
        if qmm_window_slice is None:
            num_peaks = len(peaks)
        else:
            num_peaks = 0
            start, stop = qmm_window_slice.start, qmm_window_slice.stop
            for rec in peaks:
                top_x = rec[1]
                if top_x >= start and top_x < stop:
                    num_peaks += 1
    return num_peaks

def get_denoise_rank_impl(ecurve, factor=None):
    num_peaks = get_cut_num_peaks(ecurve)
    if factor is None:
        factor = get_setting('conc_dependence')
    return min(16, num_peaks*factor + (num_peaks-1) + 2)

def get_num_components(ecurve):
    num_peaks = get_cut_num_peaks(ecurve)
    print('num_peaks=', num_peaks)
    return min(16, 8 + num_peaks*2)

WAVELENGTH_POINTS = 280, 260
ANGLE_POINTS = 0.02, 0.04

class E11nInfo:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

class PairedDataSets:
    def __init__(self, in_folder, kwargs):
        self.logger = logging.getLogger(__name__)
        lpm_correct = kwargs.pop('lpm_correct', False)

        if lpm_correct:
            from molass_legacy.DataStructure.CorrectedData import CorrectedXray, CorrectedUv
            rx = CorrectedXray(in_folder)
            try:
                ru = CorrectedUv(in_folder)
            except AssertionError:
                ru = rx.make_uv_proxy()
        else:
            from molass_legacy.DataStructure.RawData import RawXray, RawUv
            rx = RawXray(in_folder)
            try:
                ru = RawUv(in_folder)
            except AssertionError:
                ru = rx.make_uv_proxy()

        self.kwargs = kwargs
        self.pair = (ru, rx)

    def get_i_pos_values(self, dtype):
        values = WAVELENGTH_POINTS if dtype == 'UV' else ANGLE_POINTS
        return values

    def generate_sample_datasets(self, quad=False, bubble_care=False):
        kwargs = self.kwargs
        ru, rx = self.pair

        lpm_2d = kwargs.pop('lpm_2d', False)
        smoothing = kwargs.get('smoothing', False)

        i = ru.get_row_index(WAVELENGTH_POINTS[0])
        y1 = ru.data[i,:]
        if lpm_2d:
            from molass_legacy.DataStructure.LPM import get_corrected
            y1 = get_corrected(y1)

        i = rx.get_row_index(ANGLE_POINTS[0])
        y2 = rx.data[i,:]
        if bubble_care:
            raw_xr = rx.xr
            raw_y2 = raw_xr.data[i,:]
            from molass_legacy.SerialAnalyzer.AbnormalityCheck import bubble_check_impl
            exclude = bubble_check_impl(raw_y2)
            print('generate_sample_datasets: exclude=', exclude)
            if len(exclude) > 0:
                raw_xr.exclude_elution(exclude)
                self.logger.warning("raw elution %s have been excluded and interpolated.", str(exclude))
                rx.initialize(raw_xr)
                y2 = rx.data[i,:]

        if lpm_2d:
            from molass_legacy.DataStructure.LPM import get_corrected
            y2 = get_corrected(y2)

        if smoothing:
            from molass_legacy.KekLib.SciPyCookbook import smooth
            y1 = smooth(y1)
            y2 = smooth(y2)

        data1 = generate_sample_data(y1, 2)
        data2 = generate_sample_data(y2, 2)

        if not quad:
            return y1, y2, data1, data2

        i = ru.get_row_index(WAVELENGTH_POINTS[1])
        y3 = ru.data[i,:]
        if lpm_2d:
            from molass_legacy.DataStructure.LPM import get_corrected
            y3 = get_corrected(y3)

        i = rx.get_row_index(ANGLE_POINTS[1])
        y4 = rx.data[i,:]
        if lpm_2d:
            from molass_legacy.DataStructure.LPM import get_corrected
            y4 = get_corrected(y4)

        if smoothing:
            from molass_legacy.KekLib.SciPyCookbook import smooth
            y3 = smooth(y3)
            y4 = smooth(y4)

        data3 = generate_sample_data(y3, 2)
        data4 = generate_sample_data(y4, 2)

        return [y1, y3, y2, y4], [data1, data3, data2, data4]

    def guess_denoise_rank(self, ipe):
        forced_denoise_rank = get_setting('forced_denoise_rank')
        if forced_denoise_rank is None:
            try:
                denoise_rank = get_denoise_rank_impl(self.pair[1].ecurve)
            except:
                denoise_rank = 10
        else:
            denoise_rank = forced_denoise_rank
        self.logger.info("denoise_rank is assumed to be %d", denoise_rank)
        return denoise_rank

    def guess_denoise_rank_try(self, ipe):
        return 21 if ipe else 11

    def draw_exprapolated(self, ano, ax, argC, size=None, ipe=False):
        ux = 0 if ano < 2 else 1

        if size is None:
            size = argC.shape[0]

        if ux == 0:
            x = self.pair[ux].wvector
            ax.set_xlim(200, 300)
            C = argC
        else:
            x = self.pair[ux].qvector
            ax.set_xlim(0, 0.25)
            if ipe:
                C = np.vstack([argC, argC**2])
            else:
                C = argC

        rank = self.guess_denoise_rank(ipe)
        M = get_denoised_data(self.pair[ux].data, rank=rank)
        Cinv = np.linalg.pinv(C)
        P = np.dot(M, Cinv)

        for k in range(2):
            ax.plot(-10, 0, 'o')

        lines = []
        ylist = []
        for k, py in enumerate(P.T[:size]):
            y_ = py if ux == 0 else np.log10(py)
            line, = ax.plot(x, y_, label='c-%s' % num_to_char(k))
            lines.append(line)
            ylist.append(y_)
        ax.legend(loc='upper right')
        v = self.pair[ux].vector
        E = self.pair[ux].error
        return lines, np.array(ylist), E11nInfo(M=M, C=C, Cinv=Cinv, P=P, E=E, v=v, size=size)

    def update_exprapolated(self, ano, lines, C, ipe=False):
        ux = 0 if ano < 2 else 1
        rank = self.guess_denoise_rank(ipe)
        M = get_denoised_data(self.pair[ux].data, rank=rank)
        Cinv = np.linalg.pinv(C)
        P = np.dot(M, Cinv)
        for k, py in enumerate(P.T):
            y_ = py if ux == 0 else np.log10(py)
            lines[k].set_ydata(y_)
