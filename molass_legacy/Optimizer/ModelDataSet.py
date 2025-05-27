# coding: utf-8
"""
    ModelDataSet.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt 
from molass_legacy.Peaks.ElutionModels import egh
from Theory.JsPedersen1997 import F1
from molass_legacy.KekLib.SciPyCookbook import smooth
from scipy.interpolate import UnivariateSpline
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from RgProcess.RgCurve import RgCurve

class DataSet:
    def __init__(self, D, E, vec, curve):
        self.D = D
        self.E = E
        self.vec = vec
        self.curve = curve

class UvSpectrum:
    def __init__(self, params=None):
        self.params = params

    def fit(self, wv, y, debug=False):
        sy = smooth(y)
        spline = UnivariateSpline(wv, sy, s=0)
        y1 = spline.derivative(1)(wv)
        y1_spline = UnivariateSpline(wv, y1, s=0)
        roots = y1_spline.roots()
        y2 = spline.derivative(2)(wv)
        y2_spline = UnivariateSpline(wv, y2, s=0)

        max_y = np.max(y)
        wanted = np.logical_and(spline(roots)/max_y > 0.05, y2_spline(roots) < 0)
        wanted_roots = roots[wanted]

        assert len(wanted_roots) >= 2
        sigma = (wanted_roots[1] - wanted_roots[0])/4

        init_params = []
        for r in wanted_roots[0:2]:
            h = spline(r)
            m = r
            s = sigma
            t = 0
            init_params.append((h, m, s, t))

        init_params = np.array(init_params)
        if debug:
            def debug_plot(params):
                plt.push()
                fig, ax = plt.subplots()
                axt = ax.twinx()
                axt.grid(False)

                ax.plot(wv, y)
                for r in wanted_roots:
                    ax.plot(r, spline(r), 'o', color='red')
                axt.plot(wv, y1, ':', color='orange')

                ty = np.zeros(len(wv))
                for (h, m, s, t) in params:
                    cy = egh(wv, h, m, s, t)
                    ty += cy
                    ax.plot(wv, cy, ':')
                ax.plot(wv, ty, ':', color='red')

                plt.show()
                plt.pop()

            # debug_plot(init_params)

        def obj_func(p):
            params = p.reshape(init_params.shape)
            ty = np.zeros(len(wv))
            for (h, m, s, t) in params:
                cy = egh(wv, h, m, s, t)
                ty += cy
            return np.sum((ty - y)**2)

        result = minimize(obj_func, init_params.flatten())
        self.params = result.x.reshape(init_params.shape)

        if debug:
            debug_plot(self.params)

    def eval(self, wv):
        y = np.zeros(len(wv))
        for (h, m, s, t) in self.params:
            y += egh(wv, h, m, s, t)
        return y

FULL_PARAMS =  np.array([
        [35, [1.0, 200, 20,  0]],
        [31, [0.2, 250, 20,  0]],
        [23, [0.8, 400, 30,  0]],
        ])

def scale_params(scale, params):
    params_ = np.array(params)
    params_[1:] *= scale
    return params_

class RgCurveProxy(RgCurve):
    def __init__(self, qv, ecurve, cy_list, min_ratio=0.03):
        from RgProcess.RgCurve import make_availability_slices
        self.x = x = ecurve.x
        self.y = y = ecurve.y
        self.x_ = x
        self.y_ = y
        rg_ = np.zeros(len(self.x))
        max_y = np.max(y)
        slices, states = make_availability_slices(y, max_y=max_y, min_ratio=min_ratio)

        for r, cy in zip(FULL_PARAMS[:,0], cy_list):
            rg_ += r*cy/y

        self.rg_ = rg_
        self.ecurve = ecurve
        self.slices = slices
        self.states = states
        self.segments = self.create_segments(x, y, rg_, slices, states)
        self.X = None

    def create_segments(self, x, y, rg_, slices, states):
        from molass_legacy.KekLib.SciPyCookbook import smooth
        segments = []
        rg_buffer = np.zeros(len(x))
        for slice_, state in zip(slices, states):
            if state == 0:
                continue

            segments.append((x[slice_], y[slice_], rg_[slice_]))
        return segments

    def get_curve_segments(self):
        return self.segments

def generate_model_dsets(scale=1, drift_type=None, debug=False):

    xe_size = int(600*scale)
    xa_size = int(500*scale)
    x = np.arange(xe_size)
    y = np.zeros(len(x))
    qv = np.linspace(0.005, 0.5, xa_size)

    rg_list = []
    cy_list = []
    p_list = []

    def compute_p(rg):
        r = np.sqrt(5/3)*rg
        p = F1(qv, r)**2
        return p

    for rg, params in FULL_PARAMS:
        params_ = scale_params(scale, params)
        cy = egh(x, *params_)
        cy_list.append(cy)
        y += cy
        rg_list.append(rg)
        p = compute_p(rg)
        p_list.append(p)

    if drift_type is None:
        by = np.zeros(len(x))
    else:
        by = x * 0.0002 + 0.01
        rg_ = np.average(FULL_PARAMS[:,0])
        bp = compute_p(rg_)
        p_list += [bp]
        cy_list += [by]

    P = np.array(p_list).T
    C = np.array(cy_list)
    D = P @ C

    (a, b) = (2, 0)
    uv_ends = [x_*a + b for x_ in x[[0,-1]]]
    uv_x = np.linspace(*uv_ends, xe_size*a)
    uv_y = np.zeros(len(uv_x))
    uv_scales = [0.8, 1.0, 1.5]

    uv_cy_list = []
    for uv_scale, (h, m, s, t) in zip(uv_scales, FULL_PARAMS[:,1]):
        m_ = m*scale*a + b
        s_ = s*scale*a
        t_ = t*scale*a
        uv_cy = uv_scale * egh(uv_x, h, m_, s_, t_)
        uv_y += uv_cy
        uv_cy_list.append(uv_cy)

    if drift_type is None:
        uv_by = np.zeros(len(uv_x))
    else:
        uv_by = uv_x * 0.0001 + 0.01
        uv_cy_list += [uv_by]

    UvC = np.array(uv_cy_list)

    #  these params were got from extract_absorption_components("OA_Ald")
    uv_component_params= [
        [[  8.39752249, 218.24631571,  12.30526406,  -4.41740341],
         [  0.9994718,  277.66141429,  14.54344494,  -9.51796811]],
        [[ 10.33094092, 214.09172376,  12.63850912,  -1.00524139],
         [  1.05349782, 271.9127514,   17.12164758,   6.28261841]],
        [[  8.54545972, 219.34075863,  12.410232,    -5.0363978 ],
         [  0.99559913, 276.46754992,  16.58275676, -12.83937896]],
        ]

    uw_size = int(300*scale)
    wv = np.linspace(200, 450, uw_size)

    uv_p_list = []
    for uvs_params in uv_component_params:
        uvs = UvSpectrum(uvs_params)
        uv_p_list.append(uvs.eval(wv))

    if drift_type is None:
        pass
    else:
        uvs_params_ = np.average(uv_component_params, axis=0)
        uvs_ = UvSpectrum(uvs_params_)
        uv_bp = uvs_.eval(wv)
        uv_p_list += [uv_bp]

    UvP = np.array(uv_p_list).T
    UvD = UvP @ UvC

    if debug:
        from matplotlib.gridspec import GridSpec
        from MatrixData import simple_plot_3d
        plt.push()
        fig = plt.figure(figsize=(21,11))
        gs = GridSpec(2,3)

        ax1 = fig.add_subplot(gs[0,0], projection='3d')
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        ax3.set_yscale('log')
        ax4 = fig.add_subplot(gs[1,0], projection='3d')
        ax5 = fig.add_subplot(gs[1,1])
        ax6 = fig.add_subplot(gs[1,2])

        fig.suptitle("Model Data with Linear Baseline Drift", fontsize=30)
        ax1.set_title("Xray Apparant 3D View", fontsize=20)
        ax2.set_title("Xray Apparant Elution with Components", fontsize=20)
        ax3.set_title("Xray Component Scattering Curves", fontsize=20)
        ax4.set_title("UV Apparant 3D View", fontsize=20)
        ax5.set_title("UV Apparant Elution with Components", fontsize=20)
        ax6.set_title("UV Component Spectrum Curves", fontsize=20)

        simple_plot_3d(ax1, D, x=qv)

        ax2.plot(x, by + y, '-', color='orange')
        stop = None if drift_type is None else -1
        for cy in cy_list[0:stop]:
            ax2.plot(x, by + cy, ':')

        if drift_type is not None:
            ax2.plot(x, by, ':', color='red')

        for p in p_list[0:stop]:
            ax3.plot(qv, p, ':')

        if drift_type is not None:
            ax3.plot(qv, bp, ':', color='red')

        simple_plot_3d(ax4, UvD, x=wv)

        ax5.plot(uv_x, uv_by + uv_y, '-', color='blue')
        for cy in uv_cy_list[0:stop]:
            ax5.plot(uv_x, uv_by + cy, ':')

        if drift_type is not None:
            ax5.plot(uv_x, uv_by, ':', color='red')

        for uv_p in uv_p_list[0:stop]:
            ax6.plot(wv, uv_p, ':')

        if drift_type is not None:
            ax6.plot(wv, uv_bp, ':', color='red')

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
        plt.pop()

    ey = y + by
    uv_ey = uv_y + uv_by

    xr_curve = ElutionCurve(ey)
    rg_curve = RgCurveProxy(qv, xr_curve, cy_list)
    uv_curve = ElutionCurve(uv_ey)
    return (xr_curve, D), rg_curve, (uv_curve, UvD)

def extract_absorption_components(in_folder):
    from molass_legacy._MOLASS.SerialSettings import get_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from MatrixData import simple_plot_3d
    from molass_legacy.ElutionDecomposer import ElutionDecomposer

    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_corrected_sd()
    U, _, wv, uv_curve = sd.get_uv_data_separate_ly()

    print("U.shape=", U.shape, "wv.shape=", wv.shape)

    x = uv_curve.x
    y = uv_curve.y
    decomposer = ElutionDecomposer(uv_curve, x, y, deeply=True)
    opt_recs = decomposer.fit_recs

    cy_list = []
    for k, rec in enumerate(opt_recs):
        if k not in [1, 2, 4]:
            continue
        kno = rec.kno
        f = rec.evaluator
        cy = f(x)
        cy_list.append(cy)

    C = np.array(cy_list)
    P = U @ np.linalg.pinv(C)
    spectra_list = []
    for p in P.T:
        uvs = UvSpectrum()
        uvs.fit(wv, p, debug=False)
        print("uvs.params=", uvs.params)
        spectra_list.append(uvs)

    plt.push()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))
    ax1.plot(x, y, color='gray', alpha=0.3)
    for k, cy in enumerate(cy_list):
        ax1.plot(x, cy, ':', label="component-%d" % k)
    ax1.legend()

    for k, uvs in enumerate(spectra_list):
        ax2.plot(wv, uvs.eval(wv), label="component-%d" % k)
    ax2.legend()

    fig.tight_layout()
    plt.show()
    plt.pop()
