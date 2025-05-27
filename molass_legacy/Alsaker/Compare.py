"""
    Alsaker.Bridge.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
from time import time
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from molass_legacy.ATSAS.AutorgRunner import AutorgRunner

VERY_SMALL_VALUE = 1e-10

def compare_bridge_impl(caller, corrected_sd=None):
    from importlib import reload
    import molass_legacy.Alsaker.Bridge
    reload(molass_legacy.Alsaker.Bridge)
    from molass_legacy.Alsaker.Bridge import RgcodeBrige, do_it_in_the_context

    logger = logging.getLogger(__name__)
    print("compare_bridge_impl")
    if corrected_sd is None:
        sd = caller.serial_data
    else:
        sd = corrected_sd
    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    x = xr_curve.x
    y = xr_curve.y

    D[D < VERY_SMALL_VALUE] = VERY_SMALL_VALUE

    compute_all_points = True
    check_peak_tops = False
    check_y3 = False

    def compute_molass_rgs():
        sg_results = []
        t0 = time()
        for j in range(D.shape[1]):
            if j%10 == 0:
                print([j])
            data = np.array([qv, D[:,j], E[:,j]]).T
            try:
                sg = SimpleGuinier(data)
                sq_result = (sg.Rg, sg.basic_quality)
            except:
                sq_result = (np.nan, np.nan)
            sg_results.append(sq_result)
        sg_array = np.array(sg_results)
        print("SG time elapsed: %f" % (time() - t0))
        return sg_array

    def compute_atsas_rgs():
        at = AutorgRunner()
        at_results = []
        t0 = time()
        for j in range(D.shape[1]):
            if j%10 == 0:
                print([j])
            data = np.array([qv, D[:,j], E[:,j]]).T
            orig_result, _ = at.run_from_array(data)
            at_results.append((orig_result.Rg, orig_result.Rg_stdev))
        at_array = np.array(at_results)
        print("AT time elapsed: %f" % (time() - t0))
        return at_array

    def compute_peaktop_rgs(rb):
        data_list = []
        if check_y3:
            y3_list = []

        for rec in xr_curve.peak_info:
            j = rec[1]
            print([j])
            dv = D[:,j]
            ev = E[:,j]
            data = np.array([qv, dv, ev]).T
            data_list.append(data)
            where_negative = np.where(dv <= 0)[0]
            if len(where_negative) == 0:
                starting_value = None
            else:
                starting_value = where_negative[-1] + 1
            print([j], "starting_value", starting_value)
            if check_y3:
                print([j], "np.where(np.isnan(data[:,1])", np.where(np.isnan(data[:,1]))[0])
                y3 = rb.estimate_Rg(data, 1, starting_value=starting_value, return_y3=True)
                print([j], "np.where(np.isnan(y3)", np.where(np.isnan(y3))[0])
                y3_list.append(y3)
            else:
                try:
                    output = rb.estimate_Rg(data, 1, starting_value=starting_value)
                    print([j], starting_value, output)
                except:
                    log_exception(logger, "estimate_Rg failed")

        with plt.Dp():
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
            ax1.plot(x, y)
            for rec in xr_curve.peak_info:
                j = rec[1]
                ax1.plot(x[j], y[j], "o", color="red")

            for data in data_list:
                qv_, dv, ev = data.T
                ax2.plot(qv_, dv)
            if check_y3:
                for y3 in y3_list:
                    ax3.plot(y3)
            fig.tight_layout()
            plt.show()

    def compute_alsaker_rgs(rb):
        t0 = time()
        al_results = []
        for j in range(D.shape[1]):
            if j%10 == 0:
                print([j])
            data = np.array([qv, D[:,j], E[:,j]]).T
            try:
                output = rb.estimate_Rg(data, 1)
            except:
                output = (np.nan, np.nan)
            al_results.append((output[0], output[1]))
        al_array = np.array(al_results)
        print("AL time elapsed: %f" % (time() - t0))
        return al_array

    if compute_all_points:
        sg_array = compute_molass_rgs()
        at_array = compute_atsas_rgs()

    def closure():
        rb = RgcodeBrige()

        if check_peak_tops:
            compute_peaktop_rgs(rb)

        if compute_all_points:
            al_array = compute_alsaker_rgs(rb)
            return sg_array, at_array, al_array

    return do_it_in_the_context(closure)
