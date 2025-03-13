"""
    BaselineGuinier.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""

import logging
import numpy as np
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, INTEGRAL_BASELINE, restore_default_setting

BASELINE_TYPE_NAMES = ["no correction", "linear", None, None, None, "integral"]
BASELINE_TYPE_CODES = [0, 1, 5]

class GuinierResult:
    def __init__(self, sg):
        try:
            self.guinier_start = sg.guinier_start
        except:
            pass
        try:
            self.guinier_stop = sg.guinier_stop
        except:
            pass
        try:
            self.basic_quality = sg.basic_quality
        except:
            pass

class BaselineGuinier:
    def __init__(self, D, E, q, i_index, e_slice, end_slices=None):
        from molass_legacy.DataStructure.LPM import LPM_3d      # moved due to ImportError: cannot import name 'LPM_3d' from partially initialized module 'LPM' (most likely due to a circular import) 

        self.logger = logging.getLogger(__name__)

        z = D[i_index,:]
        yp = np.average(D[:,e_slice], axis=1)
        ype = np.average(E[:,e_slice], axis=1)

        lpm1 = LPM_3d(D, ecurve_y=z, integral=False, for_all_q=True, e_index=i_index)
        yp_lpm1 = np.average(lpm1.data[:,e_slice], axis=1)

        lpm2 = LPM_3d(D, ecurve_y=z, integral=True, for_all_q=False, e_index=i_index, end_slices=end_slices)
        yp_lpm2 = np.average(lpm2.data[:,e_slice], axis=1)

        curve0 = np.array([q, yp, ype]).T
        curve1 = np.array([q, yp_lpm1, ype]).T
        curve2 = np.array([q, yp_lpm2, ype]).T
        sg0 = SimpleGuinier(curve0)
        sg1 = SimpleGuinier(curve1)
        sg2 = SimpleGuinier(curve2)

        try:
            guinier_length0 = q[sg0.guinier_stop] - q[sg0.guinier_start]
        except:
            guinier_length0 = 0
        try:
            guinier_length1 = q[sg1.guinier_stop] - q[sg1.guinier_start]
        except:
            guinier_length1 = 0
        try:
            guinier_length2 = q[sg2.guinier_stop] - q[sg2.guinier_start]
        except:
            guinier_length2 = 0
        self.logger.info("lengths of guinier regions are %.3g, %.3g, %.3g", guinier_length0, guinier_length1, guinier_length2)

        self.curves = [curve0, curve1, curve2]
        self.results = [GuinierResult(sg) for sg in [sg0, sg1, sg2]]    # return proxies
        self.lengths = [guinier_length0, guinier_length1, guinier_length2]

    def get_scattering_curves(self):
        return self.curves

    def get_guinier_results(self):
        return self.results

    def get_guinier_lengths(self):
        return self.lengths

    def get_xray_baseline_type(self):
        return BASELINE_TYPE_CODES[np.argmax(self.lengths)]

    def get_xray_baseline_type_second(self):
        # https://stackoverflow.com/questions/33181350/quickest-way-to-find-the-nth-largest-value-in-a-numpy-matrix/43171216
        return BASELINE_TYPE_CODES[np.argpartition(self.lengths, -2)[-2]]

def evaluate_descending_side_upturn(sd, debug=False):
    D, E, qv, ecurve = sd.get_xr_data_separate_ly()
    small_angle = slice(0, sd.xray_index)
    peak_info = ecurve.peak_info
    slice_pair = [slice(0, peak_info[0][0]), slice(peak_info[-1][-1], None)]
    left_average = np.average(D[small_angle,slice_pair[0]])
    right_average = np.average(D[small_angle,slice_pair[1]])
    upratio = right_average/left_average

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        from molass_legacy.DataStructure.MatrixData import simple_plot_3d
        print("upratio=", upratio)
        x = ecurve.x
        y = ecurve.y
        plt.push()
        fig = plt.figure(figsize=(14,7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        simple_plot_3d(ax1, D)
        ax2.plot(x, y)
        for slice_ in slice_pair:
            ax2.plot(x[slice_], y[slice_], color='yellow')
        fig.tight_layout()
        plt.show()
        plt.pop()

    return upratio

HW = 5
def get_peaktop_slice(topx):
    return slice(topx-HW, topx+HW)

def create_bg(sd, fc, logger, debug=False):
    D_, E_, qv, e_curve = sd.get_xr_data_separate_ly()

    fc_slice = slice(*fc.get_mapped_flow_changes())     # this slice can be, or better be, replaced with slice(None, None)
                                                        # because the effect in the Xray side is always ignorable.
    i_slice = slice(0, None)
    j_slice = fc_slice
    D = D_[i_slice,j_slice]
    E = E_[i_slice,j_slice]
    q = qv[i_slice]

    end_slices = get_setting('manual_end_slices')
    if end_slices is None:
        # make sure to be clear about j0 handling
        end_slices = e_curve.get_end_slices()

    i_index = sd.xray_index
    start = 0 if j_slice.start is None else j_slice.start
    topx = e_curve.peak_info[e_curve.primary_peak_no][1] - start

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        from molass_legacy.Elution.CurveUtils import simple_plot
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("xray_baseline_type debug")
            ax1.set_title("Mapped Flow Change")
            ax2.set_title("Preliminary Guinier Analysis Range")

            simple_plot(ax1, e_curve, color="orange", legend=False)
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim()
            for j in [j_slice.start, j_slice.stop]:
                label = None if j is None else "mapped flowchange"
                ax1.plot([j, j], [ymin, ymax], color="yellow", label=label)

            ax1.legend()

            ax2.set_yscale("log")
            y = np.average(D_[:,topx-5:topx+6], axis=1)
            ax2.plot(qv, y)

            fig.tight_layout()
            plt.show()

    bg = BaselineGuinier(D, E, q, i_index, get_peaktop_slice(topx), end_slices=end_slices)
    return bg

def change_baseline_correction_defaults(sd, fc, logger, debug=False):
    """
        note that sd is not yet trimmed.
    """

    integral = False
    if hasattr(sd, 'mtd_elution') and sd.mtd_elution is not None:
        return integral

    if fc.has_special():
        uv_baseline_type = 4
        set_setting('uv_baseline_type', uv_baseline_type)
        logger.info("uv_baseline_type=%d is preferable for special type of flow changes.", uv_baseline_type)
    else:
        uv_baseline_type = get_setting('uv_baseline_type')

    xray_baseline_type = get_setting('xray_baseline_type')

    if xray_baseline_type is None:
        bg = create_bg(sd, fc, logger, debug=debug)
        xray_baseline_type = bg.get_xray_baseline_type()
    else:
        if uv_baseline_type == 4 and xray_baseline_type == 5:
            # temporary fix for cases such as 20190309_1
            bg = create_bg(sd, fc, logger, debug=debug)

    if uv_baseline_type == 4 and xray_baseline_type == 5:
        # temporary fix for cases such as 20170304
        xray_baseline_type_second = bg.get_xray_baseline_type_second()
        logger.info("xray_baseline_type is changed from %d to %d for the special type.", xray_baseline_type, xray_baseline_type_second)
        xray_baseline_type = xray_baseline_type_second

    if xray_baseline_type == 0:
        set_setting('xray_baseline_opt', 0)
    else:
        if xray_baseline_type == 5:
            integral = True
        set_setting('xray_baseline_opt', 1)

    set_setting('xray_baseline_type', xray_baseline_type)

    logger.info("changed default baseline correction: %s (xray_baseline_type=%d)", BASELINE_TYPE_NAMES[xray_baseline_type], xray_baseline_type)

    return integral
