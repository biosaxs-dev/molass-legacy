# coding: utf-8
"""
    CorrectedData.py

    Copyright (c) 2020-2021, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from RawData import RawXray, RawUv
from Prob.ProbData import ProbData
from LPM import LPM_3d
from molass_legacy.Baseline.LambertBeer import BasePlane
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Trimming.PreliminaryRg import get_default_angle_range_impl

N_SIGMA_TO_RISTRICT_RANGE = 4

class CorrectedXray(RawXray):
    def __init__(self, in_folder=None, xr=None, mf_adjust=True, average_width=None):
        self.logger = logging.getLogger(__name__)
        if xr is None:
            xr = RawXray(in_folder)
        self.initialize(xr, mf_adjust, average_width)

    def exclude_elution(self, points):
        """
        do not use this
        use RawXray.exclude_elution instead
        """
        assert False

    def initialize(self, xr, mf_adjust=True, average_width=None):
        self.xr = xr
        i = xr.get_row_index(0.02)
        eslice, xr_y = self.get_elution_range(xr, i, average_width)
        y = xr_y[eslice]
        ecurve = ElutionCurve(y, j0=xr.j0)
        aslice = self.get_angular_range(xr, eslice, ecurve)

        self.qvector = xr.qvector[aslice].copy()
        self.vector = self.qvector
        self.ecurve = ecurve
        restricted = xr.data[aslice,eslice].copy()
        lpm = LPM_3d(restricted)
        if mf_adjust:
            data = self.adjust_with_mf_baseplane(lpm.data, i, ecurve)
        else:
            data = lpm.data
        error = xr.error[aslice,eslice].copy()
        j0 = xr.j0
        self.set_data(slice(j0+eslice.start, j0+eslice.stop), data, error)
        self.excluded = xr.excluded

    def get_elution_range(self, xr, i, average_width=None):
        if average_width is None:
            average_width = get_setting('num_points_intensity')

        hw = average_width//2
        xr_y = np.average(xr.data[i-hw:i+hw+1,:], axis=0)
        prb = ProbData(xr_y)
        # prb.proof_plot()
        start, stop = prb.get_approx_peak_range(N_SIGMA_TO_RISTRICT_RANGE)
        return slice(start, stop), xr_y

    def get_angular_range(self, xr, eslice, ecurve):
        qlimit = xr.get_row_index()
        start, stop, pre_rg = get_default_angle_range_impl(xr.data[:,eslice], xr.error[:,eslice], ecurve, xr.qvector, qlimit, self.logger)
        return slice(start, stop)

    def adjust_with_mf_baseplane(self, data, index, ecurve):
        try:
            bp = BasePlane(data, index, ecurve, denoise=False)
            bp.solve()
            BP = bp.get_baseplane()
            data -= BP
        except:
            import logging
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            logger = logging.getLogger(__name__)
            etb = ExceptionTracebacker()
            logger.warning("determining a BasePlane failed with %s", str(etb))
        return data

    def set_data(self, eslice, data, error):
        slice_ = get_setting('qmm_window_slice')
        if slice_ is None:
            data_ = data
            error_ = error
            j0 = eslice.start
        else:
            start = max(eslice.start, slice_.start)
            stop = min(eslice.stop, slice_.stop)
            start_ = start - eslice.start
            stop_ = stop - eslice.start
            data_ = data[:, start_:stop_]
            error_ = error[:, start_:stop_]
            j0 = start
            self.cut_ecurve(start_, stop_)

        self.data = data_
        self.error = error_
        self.j0 = j0

    def cut_ecurve(self, start, stop):
        pass

class CorrectedUv(RawUv):
    def __init__(self, in_folder=None, uv=None, mf_adjust=True):
        if uv is None:
            uv = RawUv(in_folder)
            assert uv.data is not None
        self.uv = uv

        i = uv.get_row_index(280)
        uv_y = uv.data[i,:]
        prb = ProbData(uv_y)
        # prb.proof_plot()
        start, stop = prb.get_approx_peak_range(N_SIGMA_TO_RISTRICT_RANGE)
        eslice = slice(start, stop)

        self.wvector = uv.wvector.copy()
        self.vector = self.wvector
        self.ecurve = None
        restricted = uv.data[:,eslice].copy()
        lpm = LPM_3d(restricted)
        if mf_adjust:
            data = self.adjust_with_mf_baseplane(lpm.data, i, uv_y[eslice])
        else:
            data = lpm.data
        j0 = uv.j0
        self.set_data(slice(j0+eslice.start, j0+eslice.stop), data)
        self.error = None

    def adjust_with_mf_baseplane(self, data, index, y):
        try:
            ecurve = ElutionCurve(y, j0=self.uv.j0)
            self.ecurve = ecurve
            bp = BasePlane(data, index, ecurve, denoise=False)
            bp.solve()
            BP = bp.get_baseplane()
            data -= BP
        except:
            import logging
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            logger = logging.getLogger(__name__)
            etb = ExceptionTracebacker()
            logger.warning("determining a BasePlane failed with %s", str(etb))
        return data

    def set_data(self, eslice, data):
        slice_ = get_setting('qmm_window_slice_uv')
        if slice_ is None:
            data_ = data
            j0 = eslice.start
        else:
            start = max(eslice.start, slice_.start)
            stop = min(eslice.stop, slice_.stop)
            start_ = start - eslice.start
            stop_ = stop - eslice.start
            data_ = data[:, start_:stop_]
            j0 = start

        self.data = data_
        self.j0 = j0
