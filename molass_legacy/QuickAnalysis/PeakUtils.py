"""
    QuickAnalysis.PeakUtils.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
from .ModeledPeaks import *         # for backward compatibility

def demo_modeled_peaks(in_folder):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.Elution.CurveUtils import simple_plot

    sp = StandardProcedure()
    sd_ = sp.load_old_way(in_folder)
    pre_recog = PreliminaryRecognition(sd_)
    sd = sd_._get_analysis_copy_impl(pre_recog)
    uv_x, uv_y, xr_x, xr_y, details = get_curve_xy_impl(sd, baseline_type=1)
    uv_peaks, xr_peaks, a, b = get_modeled_peaks_impl(uv_x, uv_y, xr_x, xr_y, num_peaks=5, debug=True)

def demo_modeled_peaks_dialog(root, in_folder):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.Elution.CurveUtils import simple_plot
    from importlib import reload
    import QuickAnalysis.ModeledPeaksTester
    reload(QuickAnalysis.ModeledPeaksTester)
    from molass_legacy.QuickAnalysis.ModeledPeaksTester import TestDialog

    sp = StandardProcedure()
    sd_ = sp.load_old_way(in_folder)
    pre_recog = PreliminaryRecognition(sd_)
    sd = sd_._get_analysis_copy_impl(pre_recog)
    dialog = TestDialog(root, sd)
    dialog.show()

def guess_peak_params(xr_curve, pos, j, adj_params):
    x = xr_curve.x
    y = xr_curve.y
    spline = xr_curve.spline
    h, m, s, t = adj_params

    def objective(p):
        n_h, n_s, n_t, a_h, a_s, a_t = p
        ty = egh(x, n_h, pos, n_s, n_t) + egh(x, a_h, m, a_s, a_t)
        ry = ty - y
        return np.sum(ry[x < m]**2)

    i_h = spline(pos)
    ret = minimize(objective, (i_h, 0.5*s, 0, h, s, 0))
    n_h, n_s, n_t, a_h, a_s, a_t = ret.x
    return [(n_h, pos, n_s, n_t), (a_h, m, a_s, a_t)]

class DataFilter:
    def __init__(self, names):
        self.names = names

    def ok(self, in_folder):
        for name in self.names:
            if in_folder.find(name) >= 0:
                return True
        return False

class ModelCurve:
    def __init__(self, h, mu, sigma, tau):
        self.params = (h, mu, sigma, tau)

    def __call__(self, x):
        if np.isscalar(x):
            return egh(np.array([x]), *self.params)[0]
        else:
            return egh(x, *self.params)
