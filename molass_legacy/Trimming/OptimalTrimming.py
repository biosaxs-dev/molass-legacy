"""
    Trimming.OptimalTrimming.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import logging
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.UV.UvPreRecog import UvPreRecog

MARING_RATIO = 0.3      # relative to the peak region width
INCLUDING_FC_POS = -0.1

def get_start_stop_from_list(restrict_list, size):
    if restrict_list is None:
        e_start = 0
        e_stop = size
    else:
        etrim = restrict_list[0]
        if etrim is None:
            e_start = 0
            e_stop = size
        else:
            e_start = 0 if etrim.start is None else etrim.start
            e_stop = size if etrim.stop is None else etrim.stop
    return e_start, e_stop

def including_fc_pos(f, t):
    return f*(1+INCLUDING_FC_POS) - t*INCLUDING_FC_POS

class OptimalTrimming:
    def __init__(self, sd, pre_recog, debug=False):
        self.logger = logging.getLogger(__name__)
        if debug:
            self.sd = sd
        else:
            self.sd = None
        peak_region_width = pre_recog.cs.a_curve.get_peak_region_width()

        upr = UvPreRecog(sd, pre_recog, debug=debug)
        trim_slice = upr.get_trim_slice()
        if hasattr(sd, 'conc_array'):
            # for SerialData
            uv_size = sd.conc_array.shape[1]
        else:
            # for SecSaxs.DataSet
            uv_size = sd.uv_array.shape[1]
        ut0_ = uv_size if trim_slice.stop is None else trim_slice.stop

        uv_restrict_list = get_setting('uv_restrict_list')
        uv_e_start, uv_e_stop = get_start_stop_from_list(uv_restrict_list, uv_size)
        xr_restrict_list = get_setting('xr_restrict_list')
        xr_size = sd.intensity_array.shape[0]
        xr_e_start, xr_e_stop = get_start_stop_from_list(xr_restrict_list, xr_size)

        similarity = pre_recog.cs
        mp_e_start, mp_e_stop = [ similarity.mapped_int_value(j) for j in [xr_e_start, xr_e_stop] ]
        ut0_ = min(ut0_, mp_e_stop)

        # make sure to include flowchange point if one exists near the peak region as in 20220311
        fc = pre_recog.flowchange.get_real_flow_changes()
        fc_left = fc[0]
        if fc_left is None:
            fc_left = 0

        inc_pos = including_fc_pos(uv_e_start, uv_e_stop)
        if fc_left is None or fc_left < inc_pos:
            start = max(uv_e_start, mp_e_start)
        else:
            margin = int(peak_region_width*MARING_RATIO)
            f0 = max(0, fc_left - margin)
            start = min(f0, uv_e_start)

        peak_slice = pre_recog.cs.a_curve.peak_slice
        if (peak_slice.stop - peak_slice.start)/uv_size < 0.8:  # almost the same meaning with peak_slice.stop - peak_slice.start < uv_size
            uf0 = min(peak_slice.start, start)
            ut0 = max(peak_slice.stop, ut0_)
        else:
            uf0 = start
            ut0 = ut0_

        xf0, xt0 = [similarity.inverse_int_value(j) for j in [uf0, ut0]]
        xr_size = sd.intensity_array.shape[0]
        xf1 = max(0, xf0)
        xt1 = min(xr_size, xt0)

        self.uv_ends = [uf0, ut0]
        self.xr_ends = [xf1, xt1]
        self.base_curve_info = upr.get_base_curve_info()

        if debug:
            from molass_legacy.Elution.CurveUtils import simple_plot
            print([uf0, ut0], [xf0, xt0], [xf1, xt1])
            curves = sd.get_elution_curves()
            uv_base_curve, init_params = self.base_curve_info

            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("debug plot at OptimalTrimming")
                simple_plot(ax1, curves[0])
                x = curves[0].x
                y = curves[0].y
                ax1.plot(x, uv_base_curve(x, init_params, y), color="red")
                ymin, ymax = ax1.get_ylim()
                ax1.set_ylim(ymin, ymax)

                for j in fc:
                    if j is not None:
                        ax1.plot([j, j], [0, 0], "o", color="cyan")

                for j in [uv_e_start, uv_e_stop]:
                    ax1.plot([j, j], [ymin, ymax], ":", color="gray")

                for j in [uf0, ut0]:
                    ax1.plot([j, j], [ymin, ymax], color="yellow")

                for j in [peak_slice.start, peak_slice.stop]:
                    ax1.plot([j, j], [ymin, ymax], ":", color="green")

                simple_plot(ax2, curves[1], color="orange")
                ymin, ymax = ax2.get_ylim()
                ax2.set_ylim(ymin, ymax)
                for j in [xf1, xt1]:
                    ax2.plot([j, j], [ymin, ymax], color="yellow")
                fig.tight_layout()
                plt.show()

    def get_base_curve_info(self, debug=False):
        uv_base_curve, init_params = self.base_curve_info
        if debug and self.sd is not None:
            from molass_legacy.Elution.CurveUtils import simple_plot
            uv_curve = self.sd.get_uv_curve()
            x = uv_curve.x
            y = uv_curve.y
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("get_base_curve_info debug")

                simple_plot(ax1, uv_curve)
                by1 = uv_base_curve(x, init_params, y)
                ax1.plot(x, by1, color="red")

                simple_plot(ax2, uv_curve)
                # by2 = shifted_curve(x, shifted_params)
                # ax2.plot(x, by2, color="red")

                fig.tight_layout()
                plt.show()

        return uv_base_curve, init_params

    def set_info(self, debug=False):
        uv_restrict_list = get_setting("uv_restrict_list")
        xr_restrict_list = get_setting("xr_restrict_list")

        if debug:
            print("uv_restrict_list=", uv_restrict_list)
            print("xr_restrict_list=", xr_restrict_list)

        if uv_restrict_list is not None:
            uv_info = uv_restrict_list[0]
            if uv_info is not None:
                uv_info.start = self.uv_ends[0]
                uv_info.stop = self.uv_ends[1]

        if xr_restrict_list is not None:
            xr_info = xr_restrict_list[0]
            if xr_info is not None:
                xr_info.start = self.xr_ends[0]
                xr_info.stop = self.xr_ends[1]

        if debug:
            print("uv_restrict_list=", uv_restrict_list)
            print("xr_restrict_list=", xr_restrict_list)
        set_setting("uv_restrict_list", uv_restrict_list)
        self.logger.info("uv_restrict_list has been set to %s in OptimalTrimming", str(uv_restrict_list))
        set_setting("xr_restrict_list", xr_restrict_list)

def spike(in_folder, logger, debug=False):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition

    set_setting("in_folder", in_folder)
    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)

    pre_recog = PreliminaryRecognition(sd)
    mt = OptimalTrimming(sd, pre_recog, debug=debug)
    mt.set_info(debug=True)
