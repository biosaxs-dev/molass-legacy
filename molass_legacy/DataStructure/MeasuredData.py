"""
    MeasuredData.py

    Copyright (c) 2019-2021, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from mpl_toolkits.mplot3d import Axes3D
from molass_legacy.DataStructure.XrayData import XrayData
from molass_legacy.DataStructure.UvData import UvData
from molass_legacy.Mapping.CurveSimilarity import CurveSimilarity

class MeasuredData:
    def __init__(self, in_folder, uv_folder=None, sd=None, pre_recog=None, debug=False):

        if sd is None:
            self.xr = XrayData(in_folder)

            if uv_folder is None:
                uv_folder = in_folder

            self.uv = UvData(uv_folder)
            self.in_folder = in_folder
        else:
            self.xr = XrayData(None, sd=sd)
            self.uv = UvData(None, sd=sd)

        if debug:
            from MatrixData import simple_plot_3d
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,6), subplot_kw={'projection': '3d'})
            simple_plot_3d(ax1, self.uv.data)
            simple_plot_3d(ax2, self.xr.data)
            fig.tight_layout()
            plt.show()

        self.pre_recognize(pre_recog)

        if debug:
            self.plot()

    def plot(self, auto=False, save=False):
        plt.push()
        fig = plt.figure(figsize=(21,7))
        fig.suptitle(get_in_folder(), fontsize=16)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133)
        self.uv.plot(ax=ax1)
        self.xr.plot(ax=ax2)
        self.plot_pre_sync(ax3)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        block = not auto
        pause = 2 if auto else None
        plt.show(block=block, pause=pause)
        if save:
            import os
            from molass_legacy._MOLASS.SerialSettings import get_setting
            path = os.path.join(get_setting('analysis_folder'), get_setting('analysis_name'))
            fig.savefig(path)
        plt.pop()

    def pre_recognize(self, pre_recog):
        if pre_recog is None:
            self.cs = CurveSimilarity(self.uv.e_curve, self.xr.e_curve)
        else:
            self.cs = pre_recog.cs
        self.set_restriction(pre_recog)

    def plot_pre_sync(self, ax):

        ax.set_title("Preliminary Mapping", fontsize=16)
        x = self.xr.e_curve.x
        xr_y = self.xr.e_curve.y
        ax.plot(x, xr_y, color='orange', label='Elution Curve in Xray Scattering')

        x_ = self.cs.get_extended_x()
        uv_y = self.cs.get_uniformly_mapped_a_curve(x=x_)
        ax.plot(x_, uv_y, ':', color='blue', label='Elution Curve in UV Absorbance')
        for fx in [self.xr.j_slice.start, self.xr.j_slice.stop]:
            # print('fx=', fx, x[-1])
            if fx is not None:
                ylim = ax.get_ylim()
                ax.set_ylim()
                ax.plot([fx, fx], ylim, ':', color='yellow', linewidth=3, label='Flow Change Boundary')

        ax.legend()

    def set_restriction(self, pre_recog):
        mapping_params = (self.cs.slope, self.cs.intercept)
        if pre_recog is None:
            flow_changes = self.uv.set_restriction(self.xr.e_curve)
        else:
            flow_changes = pre_recog.get_real_flow_changes()    # should be real_flow_changes to avoid misjudgement after mapping
            # print('flow_changes=', flow_changes)
            self.uv.set_restriction(self.xr.e_curve, flow_changes)
        self.xr.set_restriction(mapping_params, flow_changes)

    def get_real_flow_changes(self):
        j_slice = self.xr.j_slice
        return j_slice.start, j_slice.stop

    def get_xdata_for_mct(self):
        return self.xr.copy_for_mct()

    def update_picking_params(self, logger):
        from molass_legacy._MOLASS.SerialSettings import get_xray_picking, get_setting, set_setting
        nh = get_setting('num_points_intensity')//2     # nh = 5 in default
        n = self.xr.i_slice.start + nh
        qvector = self.xr.vector
        q_cur = get_xray_picking()
        q_new = qvector[n]
        if q_new > q_cur:
            q_new_r = round(q_new, 2)   # e.g., 0.0286 -> 0.03
            set_setting('x_ecurve_picking_q', q_new_r)
            logger.info('ecurve-picking Q has been updated from %.3g to %.3g', q_cur, q_new_r)
            ret = True
        else:
            ret = False
        return ret

    def get_sd(self, jslice=None, xr_jslice=None, uv_jslice=None):
        from SerialDataUtils import get_xray_files, get_uv_filename
        from SerialData import SerialData
        datafiles = get_xray_files(self.in_folder)

        xr = self.xr
        qvector = xr.vector
        data = xr.data
        error = xr.error.data
        if xr_jslice is None:
            xr_jslice = jslice
        if xr_jslice is None:
            start = 0
            stop = data.shape[1]
        else:
            start = xr_jslice.start
            if start is None:
                start = 0
            stop = xr_jslice.stop
            if stop is None:
                stop = data.shape[1]
        xray_array = np.array([np.vstack([qvector, data[:,j], error[:,j]]).T for j in range(start, stop)])
        datafiles_ = datafiles[start:stop]

        uv_file = get_uv_filename(self.in_folder)
        uv = self.uv

        if uv_jslice is None:
            uv_jslice = jslice
        if uv_jslice is None:
            start = 0
            stop = uv.data.shape[1]
        else:
            start = uv_jslice.start
            stop = uv_jslice.stop
            if start is None:
                start = 0
            if stop is None:
                stop = uv.data.shape[1]

        uv_array = uv.data[:,start:stop]
        print('get_sd: uv_array.shape=', uv_array.shape)
        lvector = uv.vector
        col_header = uv.col_header[start:stop]
        mtd_elution = None
        data_info   = [ datafiles_, xray_array, uv_array, lvector, col_header, mtd_elution ]
        sd = SerialData( self.in_folder, self.in_folder, conc_file=uv_file, data_info=data_info )
        return sd

    def create_mapper(self, **kwargs):
        from molass_legacy.Mapping.ElutionMapper import ElutionMapper
        from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
        sd = self.get_sd(**kwargs)
        mapper = ElutionMapper(sd)
        absorbance = mapper.absorbance
        pre_recog = PreliminaryRecognition(sd)
        absorbance.compute_base_curve(pre_recog, 1)     # baseline type should be one of 0, 1, 4
        mapper.optimize()
        return mapper
