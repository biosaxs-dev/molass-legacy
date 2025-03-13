"""
    Guinier/RgComputer.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from molass_legacy.DataStructure.AnalysisRangeInfo import shift_range_from_to_by_x

def compute_rgs(M, E, qv, C, paired_ranges, x=None, return_trs=False, return_qualities=False, debug=False):
    assert C.shape[0] == len(paired_ranges)

    ret_rgs = []
    ret_trs = []
    peak_rgs = []
    peak_trs = []
    qualities = []
    includes_nan = False
    for k, prange in enumerate(paired_ranges):
        target_i = k
        temp_rgs = []
        temp_trs = []
        temp_qualities = []
        for f, t in prange.get_fromto_list():
            tr_ = (f+t)/2
            ret_trs.append(tr_)
            if x is not None:
                f, t = shift_range_from_to_by_x(x, f, t)
            if debug:
                print([k], f, t)
            slice_ = slice(f,t+1)
            M_ = M[:,slice_]
            C_ = C[:,slice_]
            E_ = E[:,slice_]
            P_ = M_ @ np.linalg.pinv(C_)
            Minv = np.linalg.pinv(M_)
            W = Minv @ P_
            Pe = np.sqrt(E_**2 @ W**2)
            data = np.array([qv, P_[:,target_i], Pe[:,target_i]]).T
            sg = SimpleGuinier(data)
            if sg.Rg is None or sg.Rg < 10:
                rg = np.nan
                includes_nan = True
            else:
                rg = sg.Rg
            ret_rgs.append(rg)
            temp_rgs.append(rg)
            temp_trs.append(tr_)
            temp_qualities.append(sg.basic_quality)
        peak_rgs.append(np.nanmean(temp_rgs))
        peak_trs.append(np.nanmean(temp_trs))
        qualities.append(np.nanmean(temp_qualities))

    if includes_nan:
        peak_rgs, qualities = interpolate_nan(peak_rgs, peak_trs, qualities)

    if return_trs:
        if return_qualities:
            return np.array(ret_rgs), np.array(ret_trs), np.array(peak_rgs), np.array(peak_trs), np.array(qualities)
        else:
            return np.array(ret_rgs), np.array(ret_trs), np.array(peak_rgs), np.array(peak_trs)
    else:
        return np.array(ret_rgs)

def interpolate_nan(peak_rgs, peak_trs, qualities, debug=False):
    """
    task: make this more reliable
    """
    isnan_indeces = np.where(np.isnan(peak_rgs))[0]
    if len(isnan_indeces) == 0:
        return peak_rgs, qualities

    # as in 20161104/BL-6A/pH6
    import logging
    from scipy.optimize import minimize
    from scipy.interpolate import UnivariateSpline
    logger = logging.getLogger(__name__)
    logger.info("interpolate_nan: isnan_indeces=%s", isnan_indeces)
    temp_rgs = np.asarray(peak_rgs).copy()
    temp_trs = np.asarray(peak_trs)
    temp_qualities = qualities.copy()
    valid_indeces = np.where(np.logical_not(np.isnan(temp_rgs)))[0]
    x = temp_trs[valid_indeces]
    y = temp_rgs[valid_indeces]
    """
    tr = t0 + NT*(1 - rg/poresize)**3
    """
    def objective(p):
        t0, NT, poresize = p
        rhov = x/poresize
        rhov[rhov > 1] = 1
        x_ = t0 + NT*(1 - rhov)**3
        return np.sum((x_ - x)**2)

    t0_init = np.min(x) - 30
    res = minimize(objective, (t0_init, 300, 80))
    t0, NT, poresize = res.x
    y_ = np.linspace(70, 30, 20)
    rhov = y_/poresize
    rhov[rhov > 1] = 1
    x_ = t0 + NT*(1 - rhov)**3
    spline = UnivariateSpline(x_, y_)

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        logger.info("interpolate_nan: x=%s, y=%s", x, y)

        fig, ax = plt.subplots()
        ax.set_title("interpolate_nan debug")
        ax.plot(x, y, 'o')
        ax.plot(x_, y_, ":")
        ax.plot(x_, spline(x_), ":")
        t_ = temp_trs[isnan_indeces]
        ax.plot(t_, spline(t_), "o", color="red")
        fig.tight_layout()
        plt.show()

    for i in isnan_indeces:
        rg = spline(peak_trs[i])
        temp_rgs[i] = rg
        temp_qualities[i] = 0
        logger.info("interpolate_nan: %d-th Rg has been interpolated as rg=%.3g using peak_trs[%d]=%.3g ", i, rg, i, temp_trs[i])
    return temp_rgs, temp_qualities