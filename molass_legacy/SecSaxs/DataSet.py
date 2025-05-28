"""
    SecSaxs.DataSet.py

    1) successor to SerialData
    2) used only in v2 for the time being

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import logging
import inspect
import numpy as np
from bisect import bisect_right
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SerialAnalyzer.ScatteringBaseUtil import apply_baseline_correction_impl
from molass_legacy.Elution.CurveUtils import get_xray_elution_vector
from .ElCurve import ElCurve
class DataSet:
    def __init__(self, xr_array, xr_ex, xr_qv, uv_array, uv_ex, uv_wv, xr_ey=None, uv_ey=None, sd=None, pre_recog=None):
        self.logger = logging.getLogger(__name__)
        self.xr_array = xr_array
        self.intensity_array = xr_array     # for compatibility with SerialData such as in OptDataSets.__init__
        self.xr_ex = xr_ex
        if xr_ey is not None:
            assert len(xr_ey) == len(xr_ex)
        self.xr_ey = xr_ey
        self.xr_qv = xr_qv
        self.qvector = xr_qv                # for backward compatibility in Optimizer.LrfExporter
        self.absorbance = None if sd is None else sd.absorbance
        self.uv_array = uv_array
        self.uv_ex = uv_ex
        if uv_ey is not None:
            assert len(uv_ey) == len(uv_ex)
        self.uv_ey = uv_ey
        self.uv_wv = uv_wv
        self.lvector = uv_wv                # for backward compatibility 
        self.sd = sd
        self.pre_recog = pre_recog
        self.orig_info = None       # for backward compatibility, or better be removed
        self.xr_curve = None
        self.uv_curve = None
        self.cd_slice = None
        if sd is None:
            self.xr_index = None
        else:
            try:
                self.xr_index = sd.xr_index
            except:
                self.xr_index = sd.xray_index

    def copy(self):
        ds = DataSet(
                self.xr_array.copy(),
                self.xr_ex.copy(),
                self.xr_qv.copy(),
                self.uv_array.copy(),
                self.uv_ex.copy(),
                self.uv_wv.copy(),
                self.xr_ey.copy(),
                self.uv_ey.copy(),
                )

        ds.absorbance = self.absorbance     # note that this is not a copy
        ds.xr_index = self.xr_index
        ds.sd = self.sd                     # necessary?
        return ds

    def get_copy(self):             # for backward compatibility
        return self.copy()

    def get_id_info(self):          # for backward compatibility
        baseline_corrected = True   # not sure
        return 'SD(id=%s, corrected=%s)' % (str(id(self)), str(baseline_corrected))

    def get_cd_slice(self):         # for backward compatibility, used in ExtrapolationSolver.py
        if self.cd_slice is None:
            qmax = bisect_right(self.qvector, get_setting('cd_eval_qmax'))
            self.cd_slice = slice(0, qmax)
        return self.cd_slice

    def get_xray_scale(self):       # for backward compatibility, used in LrfResultPool.py
        return self.xr_curve.max_y

    def get_xr_curve(self):
        if self.xr_curve is None:
            if self.sd is None:
                v1_curve = None
            else:
                v1_curve = self.sd.get_xray_curve()
            self.xr_curve = ElCurve(self.xr_ex, self.xr_ey, v1_curve=v1_curve)
        return self.xr_curve

    def get_xray_curve(self):
        # for backward compatibility with ElutionCurve
        return self.get_xr_curve()

    def get_uv_curve(self):
        if self.uv_curve is None:
            if self.sd is None:
                v1_curve = None
            else:            
                v1_curve = self.sd.get_uv_curve()
            self.uv_curve = ElCurve(self.uv_ex, self.uv_ey, v1_curve=v1_curve)
        return self.uv_curve

    def get_xr_data_separate_ly(self):
        D = self.xr_array[:,:,1].T
        E = self.xr_array[:,:,2].T
        xr_curve = self.get_xr_curve()
        return D, E, self.xr_qv, xr_curve

    def get_uv_data_separate_ly(self):
        D = self.uv_array
        uv_curve = self.get_uv_curve()
        return D, None, self.uv_wv, uv_curve

    def apply_baseline_correction( self, mapped_info, basic_lpm=False, progress_cb=None, return_base=False, debug_obj=None  ):
        frame = inspect.stack()[1]
        self.logger.info("apply baseline_correction called from %s(%d) with basic_lpm=%s", frame.filename, frame.lineno, str(basic_lpm))
        self.logger.info("id:%s, code_context:%s", str(id(self)),''.join(frame.code_context)[:-1])

        debug = False
        if debug:
            xr_array_copy = self.xr_array.copy()

        jvector = np.arange(self.xr_array.shape[0])
        base = apply_baseline_correction_impl(
            jvector, self.qvector, self.xr_array,
            mapped_info, basic_lpm=basic_lpm,
            index=self.xr_index,
            ecurve=self.get_xr_curve(),
            progress_cb=progress_cb,
            return_base=return_base, logger=self.logger, debug_obj=debug_obj )

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.DataStructure.MatrixData import simple_plot_3d
            qv = self.qvector

            with plt.Dp():
                ey1 = get_xray_elution_vector(qv, xr_array_copy)[0]
                ey2 = get_xray_elution_vector(qv, self.xr_array)[0]
                fig, ax = plt.subplots()
                ax.set_title("apply_baseline_correction: elution curves")
                ax.plot(ey1, label="original")
                ax.plot(ey2, label="corrected")
                ax.legend()
                fig.tight_layout()
                plt.show()

            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5), subplot_kw=dict(projection="3d"))
                fig.suptitle("apply_baseline_correction: 3d views")
                simple_plot_3d(ax1, self.xr_array[:,:,1].T, x=qv)
                simple_plot_3d(ax2, self.intensity_array[:,:,1].T, x=qv)
                fig.tight_layout()
                plt.show()

        try:
            self.set_xr_dependents(debug=debug)
        except:
            # not fatal as in the degraded case of 20161124
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "apply_baseline_correction: ")

        return base

    def get_xray_elution_vector( self ):
        xr_ey, slice_ = get_xray_elution_vector(self.qvector, self.xr_array)
        self.xr_slice = slice_
        self.xr_index = (slice_.start + slice_.stop)//2
        self.xray_index = self.xr_index     # for backward compatibility with LrfSolver.py
        return xr_ey

    def set_xr_dependents(self, debug=False):
        from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve

        self.xr_ey = self.get_xray_elution_vector()

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("set_ivector_etcetera")
                ax.plot(self.ivector, label="ivector")
                ax.legend()
                fig.tight_layout()
                plt.show()

        # self.xr_curve = ElCurve(self.xr_ex, self.xr_ey)   # ElCurve is not mature enough to do this
        self.xr_curve = ElutionCurve(self.xr_ey, x=self.xr_ex)

    def _get_analysis_copy_impl(self, given_pre_recog):
        xr_array = self.xr_array
        xr_ex = self.xr_ex
        xr_ey = self.xr_ey
        xr_qv = self.xr_qv
        uv_array = self.uv_array
        uv_ex = self.uv_ex
        uv_ey = self.uv_ey
        uv_wv = self.uv_wv

        kwargs = {}

        xr_restrict_list = get_setting('xr_restrict_list')
        if xr_restrict_list is None:
            xr_array_ = xr_array
            xr_ex_ = xr_ex
            xr_ey_ = xr_ey
            xr_qv_ = xr_qv
        else:
            # print('xr_restrict_list=', xr_restrict_list)
            elution_restrict = xr_restrict_list[0]
            if elution_restrict is None:
                eslice = slice(None, None)
            else:
                eslice = slice(elution_restrict.start, elution_restrict.stop)
            angle_restrict = xr_restrict_list[1]
            if angle_restrict is None:
                sslice = slice(None, None)
            else:
                sslice = slice(angle_restrict.start, angle_restrict.stop)
            xr_array_ = xr_array[eslice,sslice,:]
            xr_ex_ = xr_ex[eslice]
            xr_qv_ = xr_qv[sslice]
            self.logger.info('xr_array has been restricted from %s to %s' % (str(xr_array.shape), str(xr_array_.shape)))

        if xr_ey is not None:
            kwargs['xr_ey'] = xr_ey[eslice].copy()

        uv_restrict_list = get_setting('uv_restrict_list')
        if uv_restrict_list is None:
            uv_array_ = uv_array
            uv_ex_ = uv_ex
            uv_ey_ = uv_ey
            uv_wv_ = uv_wv
        else:
            elution_restrict = uv_restrict_list[0]
            if elution_restrict is None:
                eslice = slice(None, None)
            else:
                eslice = slice(elution_restrict.start, elution_restrict.stop)
            wl_restrict = uv_restrict_list[1]
            if wl_restrict is None:
                wslice = slice(None, None)
            else:
                wslice = slice(wl_restrict.start, wl_restrict.stop+1)
            uv_array_ = uv_array[wslice, eslice]
            uv_ex_ = uv_ex[eslice]
            uv_wv_ = uv_wv[wslice]
            self.logger.info('uv_array has been restricted from %s to %s' % (str(uv_array.shape), str(uv_array_.shape)))

        if uv_ey is not None:
            kwargs['uv_ey'] = uv_ey[eslice].copy()

        # return DataSet(xr_array, xr_ex, xr_qv, uv_array, uv_ex, uv_wv, xr_ey=xr_ey, uv_ey=uv_ey, sd=analysis_sd)
        analysis_copy = DataSet(xr_array_.copy(),
                                xr_ex_.copy(),
                                xr_qv_.copy(),
                                uv_array_.copy(),
                                uv_ex_.copy(),
                                uv_wv_.copy(),
                                **kwargs
                                )

        self.logger.info("sd analysis copy has been made with trimming info UV %s and Xray %s", str(uv_restrict_list), str(xr_restrict_list))
        self.logger.info('sd analysis copy %s made with xray_array.shape=%s from %s', analysis_copy.get_id_info(), str(analysis_copy.intensity_array.shape), analysis_copy.get_id_info() )

        return analysis_copy

def get_slices_from_list(trim_list):
    slice_list = []
    if trim_list is None:
        slice_list = [slice(None, None)] * 2
    else:
        for trim_info in trim_list:
            if trim_info is None:
                start = None
                stop = None
            else:
                start = trim_info.start
                stop = trim_info.stop
            slice_list.append(slice(start, stop))
    return slice_list

def get_vector_from_slice(size, slice_):
    start = slice_.start
    if start is None:
        start = 0
    stop = slice_.stop
    if stop is None:
        stop = size
    return np.arange(start, stop)

def copy_create_dataset_from_sd(original_sd, analysis_sd, debug=False):
    # task: reconsider this procedure        
    if False:
        xr_restrict_list = get_setting("xr_restrict_list")
        uv_restrict_list = get_setting("uv_restrict_list")

        xr_eslice, xr_aslice = get_slices_from_list(xr_restrict_list)
        uv_eslice, uv_wslice = get_slices_from_list(uv_restrict_list)

        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("copy_create_dataset_from_sd")
            for k, (ax, title, sd) in enumerate([(ax1, "original_sd", original_sd),(ax2, "analysis_sd", analysis_sd)]):
                ax.set_title(title)
                xr_curve = sd.get_xr_curve()
                ax.plot(xr_curve.x, xr_curve.y)
                if k == 0:
                    for j in xr_eslice.start, xr_eslice.stop:
                        if j is None:
                            continue
                        print("j=", j, xr_curve.x[[0,-1]])
                        ax.axvline(j, color="red")
            fig.tight_layout()
            plt.show()

    xr_array = analysis_sd.intensity_array.copy()
    # xr_ex = get_vector_from_slice(original_sd.intensity_array.shape[0], xr_eslice)
    try:
        xr_ex = analysis_sd.xr_ex.copy()
        xr_ey = analysis_sd.xr_ey.copy()
        xr_qv = analysis_sd.xr_qv.copy()
    except:
        xr_curve = analysis_sd.get_xr_curve()
        xr_ex = xr_curve.x.copy()
        xr_ey = xr_curve.y.copy()
        assert len(xr_ex) == len(xr_ey)
        xr_qv = analysis_sd.qvector.copy()

    try:
        uv_array = analysis_sd.uv_array.copy()
        uv_ex = analysis_sd.uv_ex.copy()
        uv_ey = analysis_sd.uv_ey.copy()
        uv_wv = analysis_sd.uv_wv.copy()
    except:
        uv_array = analysis_sd.conc_array.copy()
        # uv_ex = get_vector_from_slice(original_sd.conc_array.shape[1], uv_eslice)
        # uv_ey = analysis_sd.absorbance.a_vector
        uv_curve = analysis_sd.get_uv_curve()
        uv_ex = uv_curve.x.copy()
        uv_ey = uv_curve.y.copy()
        assert len(uv_ex) == len(uv_ey)
        uv_wv = analysis_sd.lvector.copy()

    return DataSet(xr_array, xr_ex, xr_qv, uv_array, uv_ex, uv_wv, xr_ey=xr_ey, uv_ey=uv_ey, sd=analysis_sd)

