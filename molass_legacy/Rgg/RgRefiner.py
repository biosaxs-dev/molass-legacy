# coding: utf-8
"""
    Rgg.RgRefiner.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Peaks.ElutionModels import egh_pdf

TAU_LIMIT_RATIO = 0.5   # relative to sigma
VERY_SMALL_VALUE = 1e-10
XRAY_SCALE = 1

class RgRefiner:
    def __init__(self, init_params):
        self.init_params = np.array(init_params)

    def fit(self, x, y, rg, mask=None, seed=None, min_overlap=False, debug=False):
        spline = UnivariateSpline(x, y, s=0)
        num_components = len(self.init_params)
        if mask is None:
            xm = x
            ym = y
            rgm = rg
        else:
            xm = x[mask]
            ym = y[mask]
            rgm = rg[mask]  

        max_y = np.max(ym)
        xr_scale = XRAY_SCALE
        rg_scale = max_y/np.average(self.init_params[:,4])              # average rgs
        tau_scale = max_y/np.average(self.init_params[:,2])   # average sigmas
        overlap_base = max_y*0.01

        params_zeros = np.zeros(len(self.init_params))
        debug_internal = False

        plot_history = debug
        get_overlap = False
        if plot_history:
            history_data = []

        def obj_func(p):
            nonlocal debug_internal
            p_ = p.reshape(self.init_params.shape)
            ty = np.zeros(len(xm))
            cy_list = []
            for k in range(num_components):
                w, m, s, t = p_[k,0:4]
                cy = w * egh_pdf(xm, m, s, t)
                ty += cy
                cy_list.append(cy)

            scale = max_y/np.max(ty)
            trg = np.zeros(len(xm))
            for k in range(num_components):
                crg = p_[k,4]
                trg += crg*cy_list[k]/ty

            if debug_internal:
                plt.push()
                fig, ax = plt.subplots()
                axt = ax.twinx()
                axt.grid(False)

                ax.plot(xm, ym, color='C0')
                axt.plot(xm, rgm, color='C1')
                for cy in cy_list:
                    ax.plot(xm, scale*cy, ':')
                ax.plot(xm, scale*ty, ':', color='red')

                axt.set_ylim(0, 50)
                fig.tight_layout()
                debug_internal = plt.show()
                plt.pop()

            r1 = np.sum((scale*ty - ym)**2)
            r2 = np.sum((trg - rgm)**2)

            # should be 2*tau < sigma for each tau sigma
            tau_penalty = np.sum(np.max([params_zeros, np.abs(p_[:,3]) - p_[:,2]*TAU_LIMIT_RATIO], axis=0)**2)
            tau_penalty = max(VERY_SMALL_VALUE, tau_penalty)

            resid1 = np.log(xr_scale*r1)
            resid2 = np.log(rg_scale*r2)
            penalty = np.log(tau_scale*tau_penalty)

            if min_overlap:
                overlap = np.zeros(len(xm))
                for i in range(len(cy_list)-1):
                    cy1 = cy_list[i]
                    cy2 = cy_list[i+1]
                    overlap += np.abs(np.min([cy1, cy2], axis=0))       # np.abs() is intended to degrade negative elements
                if get_overlap:
                    return scale*overlap

                if True:
                    overlap_penalty = np.log(overlap_base + scale*np.average(overlap))  # overlap_base is intended to avoid overlap_penalty getting too small
                else:
                    old_settings = np.seterr(all='raise')
                    try:
                        overlap_penalty = np.log(overlap_base + scale*np.average(overlap))  # overlap_base is intended to avoid overlap_penalty getting too small
                    except Exception as exc:
                        print("overlap=", overlap)
                        print(exc)
                        debug_internal = True
                        obj_func(p)
                    np.seterr(**old_settings)
            else:
                overlap_penalty = 0

            if plot_history:
                history_data.append((resid1, resid2, penalty, overlap_penalty))

            return resid1 + resid2 + penalty + overlap_penalty

        bounds = []
        for w, m, s, t, rg in  self.init_params:
            bounds += [(0, 10), (0, m*2), (0, s*2), (-s, s), (0.5*rg, 2*rg)]

        # result = minimize(obj_func, self.init_params.flatten(), bounds=bounds)
        if seed is None:
            seed = np.random.randint(1000, 9999)
        # minimizer_kwargs = { "bounds":bounds }
        # minimizer_kwargs = { "method": "L-BFGS-B","bounds":bounds }
        minimizer_kwargs = None
        result = basinhopping(obj_func, self.init_params.flatten(), minimizer_kwargs=minimizer_kwargs, seed=seed)
        self.params = result.x.reshape(self.init_params.shape)
        self.func_value = result.fun
        self.seed = seed
        if min_overlap:
            get_overlap = True
            self.overlap = obj_func(result.x)
        else:
            self.overlap = None
        self.xm = xm
        self.ym = ym
        self.rgm = rgm

        if debug:
            print("sigma=", self.params[:,2])
            print("tau=", self.params[:,3])
            print("rg=", self.params[:,4])
            debug_internal = True
            obj_func(result.x)
            if plot_history:
                plt.push()
                fig, ax = plt.subplots()
                ax.plot(history_data)
                fig.tight_layout()
                plt.show()
                plt.pop()

    def get_components(self, x, y, max_y=None):
        if max_y is None:
            max_y = np.max(y)
        cy_list = []
        rg_list = []
        ty = np.zeros(len(x))
        for w, m, s, t, rg in self.params[:,0:5]:
            cy = w * egh_pdf(x, m, s, t)
            ty += cy
            cy_list.append(cy)
            rg_list.append(rg)
        scale = max_y/np.max(ty)
        return [scale*cy for cy in cy_list], scale*ty, rg_list

def plot_rgrefiner_result(ax, x, y, rgc, rr, mm_seed, min_overlap=False):
    cy_list, ty, rg_list = rr.get_components(x, y)

    ax.set_title("Refined Result with seeds=(%d, %d)" % (mm_seed, rr.seed), fontsize=16)
    ax.set_ylabel('Xray Intensity')
    ax.set_xlabel('Eno')
    ax.plot(x, y, color='orange')

    for k, cy in enumerate(cy_list):
        ax.plot(x, cy, ':', label='component-%d' % (k+1))
    ax.plot(x, ty, ':', color='red', label='total')
    ax.legend(loc='upper right')

    axt = ax.twinx()
    axt.grid(False)
    rg_ = np.zeros(len(x))
    for cy, rg in zip(cy_list, rg_list):
        rg_ += rg*cy/ty

    k = 0
    for slice_, state in zip(rgc.slices, rgc.states):
        if state == 0:
            continue

        x_, y_, rg = rgc.segments[k]
        label = "observed Rg" if k == 0 else None
        axt.plot(x_, rg, ':', color='C1', label=label)
        label = "reconstructed Rg" if k == 0 else None
        axt.plot(x_, rg_[slice_], color='gray', label=label)
        k += 1

    ymin, ymax = axt.get_ylim()
    axt.set_ylim(0, ymax)
    axt.legend(loc='upper left')

    if min_overlap:
        xm = rr.xm
        y1 = np.zeros(len(xm))
        y2 = rr.overlap
        ax.fill_between(xm, y1, y2, fc='pink', alpha=0.2)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    tx = xmin*0.95 + xmax*0.05
    ty = ymin*0.5 + ymax*0.5
    ax.text(tx, ty, "fv=%.4g" % rr.func_value, fontsize=20, alpha=0.3)

def spike_demo():
    from .SpikeData import generate_demo_data, make_availability_slices, FULL_PARAMS, normal_pdf
    from .GmmSpike import spike_demo_impl as gmm_impl

    x, y, rg, num_components = generate_demo_data()
    mm = gmm_impl([(x, y, rg, None)], num_components=num_components)

    weights = mm.weights_/np.sum(mm.weights_)
    init_params = [(w, m, np.sqrt(c), 0, r) for w, m, c, r in zip(weights, mm.means_[:,0], mm.covariances_[:,0,0], mm.means_[:,1])]
    init_params = np.array(sorted(init_params, key=lambda x: x[1]))
    print("init_params=", init_params)

    rr = RgRefiner(init_params)

    max_y = np.max(y)
    slices, states = make_availability_slices(y, max_y=max_y, min_ratio=0.03)
    mask = np.zeros(len(x), dtype=bool)
    for slice_, state in zip(slices, states):
        if state:
            mask[slice_] = True

    rr.fit(x, y, rg, mask, min_overlap=True, debug=True)
