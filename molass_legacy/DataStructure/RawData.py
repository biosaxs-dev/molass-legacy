# coding: utf-8
"""
    RawData.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from SerialDataUtils import load_intensity_files, load_uv_array
from molass_legacy.KekLib.NumpyUtils import np_loadtxt
from molass_legacy._MOLASS.SerialSettings import get_setting, get_xray_picking
from MatrixData import simple_plot_3d

def get_row_index_impl(qvector, q):
    return min(len(qvector)-1, bisect_right(qvector, q))

class RawMatrix:
    def plot_3d(self, ax, alpha=1):
        simple_plot_3d(ax, self.data, x=self.vector, alpha=alpha)

class RawXray(RawMatrix):
    def __init__(self, in_folder):
        array, files = load_intensity_files(in_folder)
        assert len(array.shape) == 3
        array = self.apply_restriction(array)
        self.qvector = array[0,:,0]
        self.vector = self.qvector
        self.data = array[:,:,1].T
        self.error = array[:,:,2].T
        self.excluded = None

    def apply_restriction(self, array):
        xr_restrict_list = get_setting('xr_restrict_list')
        # print('xr_restrict_list=', xr_restrict_list)
        if xr_restrict_list is None:
            a_slice = slice(None)
            e_slice = slice(None)
            j0 = 0
        else:
            elution_info, angular_info = xr_restrict_list
            a_slice = slice(None) if angular_info is None else slice(angular_info.start, angular_info.stop)
            e_slice = slice(None) if elution_info is None else slice(elution_info.start, elution_info.stop)
            j0 = 0 if elution_info is None else elution_info.start
        self.j0 = j0
        return array[e_slice,a_slice,:].copy()

    def get_row_index(self, q=None):
        if q is None:
            q = get_xray_picking()
        return get_row_index_impl(self.qvector, q)

    def make_uv_proxy(self):
        i = self.get_row_index(0.02)
        return RawUvProxy(self.data[i,:], j0=self.j0)

    def exclude_elution(self, points):
        """
        ported
        from SerialData.exclude_intensities
        """
        from_ = None
        for i in points:
            if from_ is None:
                from_ = i
            else:
                if i > last + 1:
                    self.exclude_elution_impl(from_, last)
                    from_ = i

            last = i
        if from_ is not None:
            self.exclude_elution_impl(from_, last)
        self.excluded = points

    def exclude_elution_impl(self, from_, to_):
        """
        ported
        from SerialData.exclude_intensity
        """
        data = self.data

        debug = False
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            print('exclude_elution_impl: ', from_, to_)

            k = self.get_row_index(0.02)
            plt.push()
            plt.plot(data[k,:])
            plt.show()
            plt.pop()

        errr = self.error
        size = data.shape[1]
        if from_ == 0:
            j = to_ + 1
            for i in range(from_, j):
                if debug: print([i], 'extended')
                data[:,i] = data[:,j]
                errr[:,i] = errr[:,j]
        elif to_ == size - 1:
            j = from_ - 1
            for i in range(from_, size):
                if debug: print([i], 'extended')
                data[:,i] = data[:,j]
                errr[:,i] = errr[:,j]
        else:
            lower = from_ - 1
            upper = to_ + 1
            lower_data = data[:,lower]
            lower_errr = errr[:,lower]
            upper_data = data[:,upper]
            upper_errr = errr[:,upper]
            width = upper - lower
            for i in range(1, width):
                if debug: print([lower+i], 'interpolated with', lower, upper)
                w = i/width
                data[:,lower+i] = (1-w)*lower_data + w*upper_data
                errr[:,lower+i] = (1-w)*lower_errr + w*upper_errr

        if debug:
            k = self.get_row_index(0.02)
            plt.push()
            plt.plot(data[k,:])
            plt.show()
            plt.pop()

class RawUv(RawMatrix):
    def __init__(self, in_folder):
        array, vector, _ = load_uv_array(in_folder)
        assert array is not None
        array, vector = self.apply_restriction(array, vector)
        self.wvector = vector
        self.vector = vector
        self.data = array
        self.error = None

    def apply_restriction(self, array, vector):
        uv_restrict_list = get_setting('uv_restrict_list')
        # print('uv_restrict_list=', uv_restrict_list)
        if uv_restrict_list is None:
            w_slice = slice(None)
            e_slice = slice(None)
            j0 = 0
        else:
            elution_info, wavelen_info = uv_restrict_list
            w_slice = slice(None) if wavelen_info is None else slice(wavelen_info.start, wavelen_info.stop)
            e_slice = slice(None) if elution_info is None else slice(elution_info.start, elution_info.stop)
            j0 = 0 if elution_info is None else elution_info.start
        self.j0 = j0
        return array[w_slice, e_slice].copy(), vector[w_slice].copy()

    def get_row_index(self, w):
        return min(len(self.wvector)-1, bisect_right(self.wvector, w))

class RawUvProxy(RawUv):
    def __init__(self, ey, j0=0):
        import os
        import logging

        logger = logging.getLogger()
        logger.info("creating a RawUvProxy")

        this_dir = os.path.dirname( os.path.abspath( __file__ ) )
        ridge_data, _ = np_loadtxt(this_dir + '/../SerialAnalyzer/UV.dat')
        vector = ridge_data[:,0]
        curve_y = ridge_data[:,1]
        self.wvector = vector
        self.vector = vector
        self.data = np.dot(curve_y.reshape(len(curve_y),1), ey.reshape(1,len(ey)))
        self.error = None
        self.j0 = j0
