"""
    Peaks/LrfRgComputer.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ModelUtils import compute_cy_list, compute_area_props, get_paired_ranges_from_params_array

def compute_rgs_from_lrf_source(lrf_src, keep_num_components=False, want_num_components=None, select=None, debug=True):
    # task: unify this with Selective.MinimizerUtils.RgComputer
    # task: change get_paired_ranges_from_params_array into a class
    if debug:
        from importlib import reload
        import molass_legacy.GuinierTools.RgComputer
        reload(molass_legacy.GuinierTools.RgComputer)
    from molass_legacy.GuinierTools.RgComputer import compute_rgs

    print("compute_rgs_from_lrf_source")

    x = lrf_src.xr_x
    model = lrf_src.model
    cy_list = compute_cy_list(model, x, lrf_src.xr_peaks)
    props = compute_area_props(cy_list)
    if debug: 
        print("props=", props)

    xr_peaks = lrf_src.xr_peaks
    num_peaks = len(xr_peaks)
    M, E, qv, xr_curve = lrf_src.corrected_sd.get_xr_data_separate_ly()
    paired_ranges, indeces = get_paired_ranges_from_params_array(model, x, xr_peaks, return_indeces=True, want_num_components=want_num_components, select=select)
    C = np.array(cy_list)[indeces,:]    # note that, by indexing with "indeces", C does not include ignored components
    try:
        rgs, trs, peak_rgs, peak_trs, qualities = compute_rgs(M, E, qv, C, paired_ranges, x=x, return_trs=True, return_qualities=True, debug=debug)
    except AssertionError:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None, "compute_rgs: ")
        print("indeces=", indeces)
        print("C.shape=", C.shape)
        print("len(paired_ranges)=", len(paired_ranges))
        y = lrf_src.xr_y
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("compute_rgs AssertionError")
            ax.plot(x, y)
            for k, cy in enumerate(cy_list):
                ax.plot(x, cy, ":", label="component-%d" % k)
            for k, pr in enumerate(paired_ranges):
                for f, t in pr.get_fromto_list():
                    print([k], f, t)
                    ax.axvspan(x[f], x[t], alpha=0.3, color="C%d" % k)
            ax.legend()
            fig.tight_layout() 
            plt.show()

    if keep_num_components:
        if len(paired_ranges) < num_peaks:
            import logging
            # compute minor rgs according to the exclusion curve
            # because they could be unreliable if computed directly by LRF
            full_paired_ranges = get_paired_ranges_from_params_array(model, x, xr_peaks, ignoreable_ratio=0)
            assert len(full_paired_ranges) == num_peaks
            spline = make_exclcurve_spline(rgs, trs)
            full_indeces = np.arange(num_peaks)
            minor_indeces = np.setdiff1d(full_indeces, indeces)
            new_peak_rgs = np.zeros(num_peaks)
            new_peak_trs = np.zeros(num_peaks)
            new_qualities = np.zeros(num_peaks)
            new_peak_rgs[indeces] = peak_rgs
            new_peak_trs[indeces] = peak_trs
            new_qualities[indeces] = qualities
            for i in minor_indeces:
                paired_range = full_paired_ranges[i]
                ft_list = paired_range.get_fromto_list()
                if len(ft_list) == 1:
                    tr = x[0] + np.average(ft_list[0])
                else:
                    tr = x[0] + ft_list[0][1]
                rg = spline(tr)
                new_peak_rgs[i] = rg
                new_peak_trs[i] = tr

            logger = logging.getLogger(__name__)
            logger.warning("Minor Rgs at %s in %s have been computed according to the exclusion curve.", str(minor_indeces), str(full_indeces))

            peak_rgs = new_peak_rgs
            peak_trs = new_peak_trs
            qualities = new_qualities
            indeces = full_indeces
        
    if debug:
        print("rgs=", rgs, "trs=", trs)

    ret_props = props[indeces]
    return rgs, trs, props, peak_rgs, peak_trs, ret_props/np.sum(ret_props), indeces, qualities

def make_exclcurve_spline(rgs, trs, debug=False):
    def objective(p):
        N, T, x0, poresize = p
        rhov = rgs/poresize
        rhov[rhov > 1] = 1
        trs_ = x0 + N*T*np.power(1 - rhov, 3)
        return np.sum((trs - trs_)**2)
    
    init_params = [2000, 1, 0, 200]
    res = minimize(objective, init_params, method='Nelder-Mead')
    N, T, x0, poresize = res.x
    y = np.linspace(poresize, 10, 100)
    rhov = y/poresize   # note that rhov <= 1
    x = x0 + N*T*np.power(1 - rhov, 3)

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_xlabel("make_exclcurve_spline")
            ax.plot(x, y)
            ax.plot(trs, rgs, 'o', color='red')
            fig.tight_layout()
            plt.show()

    return UnivariateSpline(x, y, s=0, ext=3)